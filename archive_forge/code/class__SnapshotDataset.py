import multiprocessing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
class _SnapshotDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A dataset that allows saving and re-use of already processed data."""

    def __init__(self, input_dataset, path, shard_func, compression=None, reader_func=None, pending_snapshot_expiry_seconds=None, use_legacy_function=False, name=None):
        if reader_func is None:
            reader_func = lambda datasets: datasets.interleave(lambda x: x, cycle_length=multiprocessing.cpu_count(), num_parallel_calls=dataset_ops.AUTOTUNE)
        self._input_dataset = input_dataset
        self._path = path
        self._compression = compression
        self._reader_func = structured_function.StructuredFunctionWrapper(reader_func, self._transformation_name() + '.reader_func', input_structure=dataset_ops.DatasetSpec(dataset_ops.DatasetSpec(input_dataset.element_spec)), use_legacy_function=use_legacy_function)
        self._shard_func = structured_function.StructuredFunctionWrapper(shard_func, self._transformation_name() + '.shard_func', dataset=input_dataset, use_legacy_function=use_legacy_function)
        if not self._shard_func.output_structure.is_compatible_with(tensor_spec.TensorSpec([], dtypes.int32)) and (not self._shard_func.output_structure.is_compatible_with(tensor_spec.TensorSpec([], dtypes.int64))):
            raise TypeError(f'Invalid `shard_func`. `shard_func` must return `tf.int64` scalar tensor but its return type is {self._shard_func.output_structure}.')
        self._name = name
        variant_tensor = ged_ops.snapshot_dataset_v2(input_dataset._variant_tensor, path, self._reader_func.function.captured_inputs, self._shard_func.function.captured_inputs, compression=compression, reader_func=self._reader_func.function, shard_func=self._shard_func.function, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    def _functions(self):
        return [self._reader_func, self._shard_func]

    def _transformation_name(self):
        return 'Dataset.snapshot()'