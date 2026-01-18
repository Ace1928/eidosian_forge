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
def _get_worker_parts(list_of_parts: _DataParts) -> Dict[str, List[Any]]:
    assert isinstance(list_of_parts, list)
    result: Dict[str, List[Any]] = {}

    def append(i: int, name: str) -> None:
        if name in list_of_parts[i]:
            part = list_of_parts[i][name]
        else:
            part = None
        if part is not None:
            if name not in result:
                result[name] = []
            result[name].append(part)
    for i, _ in enumerate(list_of_parts):
        append(i, 'data')
        append(i, 'label')
        append(i, 'weight')
        append(i, 'base_margin')
        append(i, 'qid')
        append(i, 'label_lower_bound')
        append(i, 'label_upper_bound')
    return result