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
def _push_next(self):
    """Assign next batch workload to workers."""
    r = next(self._iter, None)
    if r is None:
        return
    async_ret = self._worker_pool.apply_async(self._worker_fn, (r, self._batchify_fn, self._dataset))
    self._data_buffer[self._sent_idx] = async_ret
    self._sent_idx += 1