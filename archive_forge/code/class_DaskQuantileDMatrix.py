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
class DaskQuantileDMatrix(DaskDMatrix):
    """A dask version of :py:class:`QuantileDMatrix`."""

    @_deprecate_positional_args
    def __init__(self, client: 'distributed.Client', data: _DataT, label: Optional[_DaskCollection]=None, *, weight: Optional[_DaskCollection]=None, base_margin: Optional[_DaskCollection]=None, missing: Optional[float]=None, silent: bool=False, feature_names: Optional[FeatureNames]=None, feature_types: Optional[Union[Any, List[Any]]]=None, max_bin: Optional[int]=None, ref: Optional[DMatrix]=None, group: Optional[_DaskCollection]=None, qid: Optional[_DaskCollection]=None, label_lower_bound: Optional[_DaskCollection]=None, label_upper_bound: Optional[_DaskCollection]=None, feature_weights: Optional[_DaskCollection]=None, enable_categorical: bool=False) -> None:
        super().__init__(client=client, data=data, label=label, weight=weight, base_margin=base_margin, group=group, qid=qid, label_lower_bound=label_lower_bound, label_upper_bound=label_upper_bound, missing=missing, silent=silent, feature_weights=feature_weights, feature_names=feature_names, feature_types=feature_types, enable_categorical=enable_categorical)
        self.max_bin = max_bin
        self.is_quantile = True
        self._ref: Optional[int] = id(ref) if ref is not None else None

    def _create_fn_args(self, worker_addr: str) -> Dict[str, Any]:
        args = super()._create_fn_args(worker_addr)
        args['max_bin'] = self.max_bin
        if self._ref is not None:
            args['ref'] = self._ref
        return args