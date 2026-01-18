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
def _prefetch_iter(self):
    if self._num_workers == 0:

        def same_process_iter():
            for batch in self._batch_sampler:
                ret = self._batchify_fn([self._dataset[idx] for idx in batch])
                if self._pin_memory:
                    ret = _as_in_context(ret, context.cpu_pinned(self._pin_device_id))
                yield ret
        return same_process_iter()
    return _MultiWorkerIter(self._worker_pool, self._batchify_fn, self._batch_sampler, pin_memory=self._pin_memory, pin_device_id=self._pin_device_id, worker_fn=_thread_worker_fn if self._thread_pool else _worker_fn, prefetch=self._prefetch, dataset=self._dataset if self._thread_pool else None, data_loader=self, timeout=self._timeout)