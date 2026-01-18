import pickle
import io
import sys
import signal
import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
from multiprocessing.pool import ThreadPool
import threading
import numpy as np
from . import sampler as _sampler
from ... import nd, context
from ...util import is_np_shape, is_np_array, set_np
from ... import numpy as _mx_np  # pylint: disable=reimported
class DataLoaderV1(object):
    """Loads data from a dataset and returns mini-batches of data.

    Parameters
    ----------
    dataset : Dataset
        Source dataset. Note that numpy and mxnet arrays can be directly used
        as a Dataset.
    batch_size : int
        Size of mini-batch.
    shuffle : bool
        Whether to shuffle the samples.
    sampler : Sampler
        The sampler to use. Either specify sampler or shuffle, not both.
    last_batch : {'keep', 'discard', 'rollover'}
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.

        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    batch_sampler : Sampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch. Defaults to `default_batchify_fn`::

            def default_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [default_batchify_fn(i) for i in data]
                else:
                    data = np.asarray(data)
                    return nd.array(data, dtype=data.dtype)

    num_workers : int, default 0
        The number of multiprocessing workers to use for data preprocessing.
    pin_memory : boolean, default False
        If ``True``, the dataloader will copy NDArrays into pinned memory
        before returning them. Copying from CPU pinned memory to GPU is faster
        than from normal CPU memory.
    pin_device_id : int, default 0
        The device id to use for allocating pinned memory if pin_memory is ``True``
    """

    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None, last_batch=None, batch_sampler=None, batchify_fn=None, num_workers=0, pin_memory=False, pin_device_id=0):
        self._dataset = dataset
        self._pin_memory = pin_memory
        self._pin_device_id = pin_device_id
        if batch_sampler is None:
            if batch_size is None:
                raise ValueError('batch_size must be specified unless batch_sampler is specified')
            if sampler is None:
                if shuffle:
                    sampler = _sampler.RandomSampler(len(dataset))
                else:
                    sampler = _sampler.SequentialSampler(len(dataset))
            elif shuffle:
                raise ValueError('shuffle must not be specified if sampler is specified')
            batch_sampler = _sampler.BatchSampler(sampler, batch_size, last_batch if last_batch else 'keep')
        elif batch_size is not None or shuffle or sampler is not None or (last_batch is not None):
            raise ValueError('batch_size, shuffle, sampler and last_batch must not be specified if batch_sampler is specified.')
        self._batch_sampler = batch_sampler
        self._num_workers = num_workers if num_workers >= 0 else 0
        if batchify_fn is None:
            if num_workers > 0:
                self._batchify_fn = default_mp_batchify_fn
            else:
                self._batchify_fn = default_batchify_fn
        else:
            self._batchify_fn = batchify_fn

    def __iter__(self):
        if self._num_workers == 0:

            def same_process_iter():
                for batch in self._batch_sampler:
                    ret = self._batchify_fn([self._dataset[idx] for idx in batch])
                    if self._pin_memory:
                        ret = _as_in_context(ret, context.cpu_pinned(self._pin_device_id))
                    yield ret
            return same_process_iter()
        return _MultiWorkerIterV1(self._num_workers, self._dataset, self._batchify_fn, self._batch_sampler, self._pin_memory, self._pin_device_id)

    def __len__(self):
        return len(self._batch_sampler)