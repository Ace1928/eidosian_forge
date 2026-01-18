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
class _MultiWorkerIter(object):
    """Internal multi-worker iterator for DataLoader."""

    def __init__(self, worker_pool, batchify_fn, batch_sampler, pin_memory=False, pin_device_id=0, worker_fn=_worker_fn, prefetch=0, dataset=None, data_loader=None, timeout=120):
        self._worker_pool = worker_pool
        self._batchify_fn = batchify_fn
        self._batch_sampler = batch_sampler
        self._data_buffer = {}
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._iter = iter(self._batch_sampler)
        self._worker_fn = worker_fn
        self._pin_memory = pin_memory
        self._pin_device_id = pin_device_id
        self._dataset = dataset
        self._data_loader = data_loader
        self._timeout = timeout
        for _ in range(prefetch):
            self._push_next()

    def __len__(self):
        return len(self._batch_sampler)

    def _push_next(self):
        """Assign next batch workload to workers."""
        r = next(self._iter, None)
        if r is None:
            return
        async_ret = self._worker_pool.apply_async(self._worker_fn, (r, self._batchify_fn, self._dataset))
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def __next__(self):
        self._push_next()
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, 'Data buffer should be empty at this moment'
            raise StopIteration
        assert self._rcvd_idx < self._sent_idx, 'rcvd_idx must be smaller than sent_idx'
        assert self._rcvd_idx in self._data_buffer, 'fatal error with _push_next, rcvd_idx missing'
        ret = self._data_buffer.pop(self._rcvd_idx)
        try:
            if self._dataset is None:
                batch = pickle.loads(ret.get(self._timeout))
            else:
                batch = ret.get(self._timeout)
            if self._pin_memory:
                batch = _as_in_context(batch, context.cpu_pinned(self._pin_device_id))
            self._rcvd_idx += 1
            return batch
        except multiprocessing.context.TimeoutError:
            msg = 'Worker timed out after {} seconds. This might be caused by \n\n            - Slow transform. Please increase timeout to allow slower data loading in each worker.\n            '.format(self._timeout)
            if not isinstance(self._worker_pool, multiprocessing.pool.ThreadPool):
                msg += '- Insufficient shared_memory if `timeout` is large enough.\n            Please consider reduce `num_workers` or increase shared_memory in system.\n            '
            print(msg)
            raise
        except Exception:
            self._worker_pool.terminate()
            raise

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self