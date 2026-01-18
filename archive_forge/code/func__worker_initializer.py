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
def _worker_initializer(dataset, active_shape, active_array):
    """Initialier for processing pool."""
    global _worker_dataset
    _worker_dataset = dataset
    set_np(shape=active_shape, array=active_array)