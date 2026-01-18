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
def _create_quantile_dmatrix(feature_names: Optional[FeatureNames], feature_types: Optional[Union[Any, List[Any]]], feature_weights: Optional[Any], missing: float, nthread: int, parts: Optional[_DataParts], max_bin: int, enable_categorical: bool, ref: Optional[DMatrix]=None) -> QuantileDMatrix:
    worker = distributed.get_worker()
    if parts is None:
        msg = f'worker {worker.address} has an empty DMatrix.'
        LOGGER.warning(msg)
        d = QuantileDMatrix(numpy.empty((0, 0)), feature_names=feature_names, feature_types=feature_types, max_bin=max_bin, ref=ref, enable_categorical=enable_categorical)
        return d
    unzipped_dict = _get_worker_parts(parts)
    it = DaskPartitionIter(**unzipped_dict, feature_types=feature_types, feature_names=feature_names, feature_weights=feature_weights)
    dmatrix = QuantileDMatrix(it, missing=missing, nthread=nthread, max_bin=max_bin, ref=ref, enable_categorical=enable_categorical)
    return dmatrix