import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
class NumpyMultiprocessingGenerator:

    def __init__(self, dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, output_signature, shuffle, batch_size, drop_remainder, num_workers):
        self.dataset = dataset
        self.cols_to_retain = cols_to_retain
        self.collate_fn = collate_fn
        self.collate_fn_args = collate_fn_args
        self.string_columns = [col for col, dtype in columns_to_np_types.items() if dtype in (np.unicode_, np.str_)]
        self.columns_to_np_types = {col: dtype if col not in self.string_columns else np.dtype('U1') for col, dtype in columns_to_np_types.items()}
        self.output_signature = output_signature
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.num_workers = num_workers
        self.columns_to_ranks = {col: int(spec.shape.rank) if col not in self.string_columns else int(spec.shape.rank) + 1 for col, spec in output_signature.items()}

    def __iter__(self):
        num_workers = min(self.num_workers, int(ceil(len(self.dataset) / self.batch_size)))
        per_worker_batches, final_batch, final_batch_worker = self.distribute_batches(self.dataset, self.batch_size, self.drop_remainder, num_workers, self.shuffle)
        ctx = get_context('spawn')
        names = []
        shape_arrays = []
        workers = []
        array_ready_events = [ctx.Event() for _ in range(num_workers)]
        array_loaded_events = [ctx.Event() for _ in range(num_workers)]
        base_args = {'dataset': self.dataset, 'cols_to_retain': self.cols_to_retain, 'collate_fn': self.collate_fn, 'collate_fn_args': self.collate_fn_args, 'columns_to_np_types': self.columns_to_np_types, 'columns_to_ranks': self.columns_to_ranks, 'string_columns': self.string_columns}
        with SharedMemoryContext() as shm_ctx:
            for i in range(num_workers):
                worker_random_id = str(uuid4())
                worker_name = f'dw_{i}_{worker_random_id}'[:10]
                names.append(worker_name)
                worker_shape_arrays = {col: shm_ctx.get_array(f'{worker_name}_{col}_shape', shape=(rank,), dtype=np.int64, create=True) for col, rank in self.columns_to_ranks.items()}
                shape_arrays.append(worker_shape_arrays)
                worker_indices = per_worker_batches[i]
                if i == final_batch_worker and final_batch is not None:
                    final_batch_arg = final_batch
                else:
                    final_batch_arg = None
                worker_kwargs = {'worker_name': worker_name, 'indices': worker_indices, 'extra_batch': final_batch_arg, 'array_ready_event': array_ready_events[i], 'array_loaded_event': array_loaded_events[i], **base_args}
                worker = ctx.Process(target=self.worker_loop, kwargs=worker_kwargs, daemon=True)
                worker.start()
                workers.append(worker)
            end_signal_received = False
            while not end_signal_received:
                for i in range(num_workers):
                    if not array_ready_events[i].wait(timeout=60):
                        raise TimeoutError('Data loading worker timed out!')
                    array_ready_events[i].clear()
                    array_shapes = shape_arrays[i]
                    if any((np.any(shape < 0) for shape in array_shapes.values())):
                        end_signal_received = True
                        break
                    with SharedMemoryContext() as batch_shm_ctx:
                        arrays = {col: batch_shm_ctx.get_array(f'{names[i]}_{col}', shape=shape, dtype=self.columns_to_np_types[col], create=False) for col, shape in array_shapes.items()}
                        arrays = {col: np.copy(arr) for col, arr in arrays.items()}
                        for string_col in self.string_columns:
                            arrays[string_col] = arrays[string_col].view(f'U{arrays[string_col].shape[-1]}').squeeze(-1)
                    yield arrays
                    array_loaded_events[i].set()
            for worker in workers:
                worker.join()

    def __call__(self):
        return self

    @staticmethod
    def worker_loop(dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, columns_to_ranks, string_columns, indices, extra_batch, worker_name, array_ready_event, array_loaded_event):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        if config.TF_AVAILABLE:
            import tensorflow as tf
        else:
            raise ImportError('Called a Tensorflow-specific function but Tensorflow is not installed.')
        tf.config.set_visible_devices([], 'GPU')

        def send_batch_to_parent(indices):
            batch = np_get_batch(indices=indices, dataset=dataset, cols_to_retain=cols_to_retain, collate_fn=collate_fn, collate_fn_args=collate_fn_args, columns_to_np_types=columns_to_np_types, return_dict=True)
            out_arrays = {}
            with SharedMemoryContext() as batch_shm_ctx:
                for col, cast_dtype in columns_to_np_types.items():
                    array = batch[col]
                    if col in string_columns:
                        array = array.view('U1').reshape(array.shape + (-1,))
                    shape_arrays[col][:] = array.shape
                    out_arrays[col] = batch_shm_ctx.get_array(f'{worker_name}_{col}', shape=array.shape, dtype=cast_dtype, create=True)
                    out_arrays[col][:] = array
                array_ready_event.set()
                array_loaded_event.wait()
                array_loaded_event.clear()
        with SharedMemoryContext() as shm_ctx:
            shape_arrays = {col: shm_ctx.get_array(f'{worker_name}_{col}_shape', shape=(rank,), dtype=np.int64, create=False) for col, rank in columns_to_ranks.items()}
            for batch in indices:
                send_batch_to_parent(batch)
            if extra_batch is not None:
                send_batch_to_parent(extra_batch)
            for col, array in shape_arrays.items():
                array[:] = -1
            array_ready_event.set()

    @staticmethod
    def distribute_batches(dataset, batch_size, drop_remainder, num_workers, shuffle):
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)
        num_samples = len(indices)
        incomplete_batch_cutoff = num_samples - num_samples % batch_size
        indices, last_incomplete_batch = np.split(indices, [incomplete_batch_cutoff])
        if drop_remainder or len(last_incomplete_batch) == 0:
            last_incomplete_batch = None
        indices = indices.reshape(-1, batch_size)
        num_batches = len(indices)
        final_batches_cutoff = num_batches - num_batches % num_workers
        indices, final_batches = np.split(indices, [final_batches_cutoff])
        indices = indices.reshape(-1, num_workers, batch_size)
        per_worker_indices = np.split(indices, indices.shape[1], axis=1)
        per_worker_indices = [np.squeeze(worker_indices, 1) for worker_indices in per_worker_indices]
        for i in range(len(final_batches)):
            per_worker_indices[i] = np.concatenate([per_worker_indices[i], final_batches[i].reshape(1, -1)], axis=0)
        if last_incomplete_batch is not None:
            incomplete_batch_worker_idx = len(final_batches)
        else:
            incomplete_batch_worker_idx = None
        return (per_worker_indices, last_incomplete_batch, incomplete_batch_worker_idx)