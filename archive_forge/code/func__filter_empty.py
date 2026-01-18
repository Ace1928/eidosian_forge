import collections
import logging
import platform
import socket
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, update_wrapper
from threading import Thread
from typing import (
import numpy
from . import collective, config
from ._typing import _T, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import DataFrame, LazyLoader, concat, lazy_isinstance
from .core import (
from .data import _is_cudf_ser, _is_cupy_array
from .sklearn import (
from .tracker import RabitTracker, get_host_ip
from .training import train as worker_train
def _filter_empty(booster: Booster, local_history: TrainingCallback.EvalsLog, is_valid: bool) -> Optional[TrainReturnT]:
    n_workers = collective.get_world_size()
    non_empty = numpy.zeros(shape=(n_workers,), dtype=numpy.int32)
    rank = collective.get_rank()
    non_empty[rank] = int(is_valid)
    non_empty = collective.allreduce(non_empty, collective.Op.SUM)
    non_empty = non_empty.astype(bool)
    ret: Optional[TrainReturnT] = {'booster': booster, 'history': local_history}
    for i in range(non_empty.size):
        if non_empty[i] and i == rank:
            return ret
        if non_empty[i]:
            return None
    raise ValueError('None of the workers can provide a valid result.')