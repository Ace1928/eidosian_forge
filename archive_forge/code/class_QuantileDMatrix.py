import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
class QuantileDMatrix(DMatrix):
    """A DMatrix variant that generates quantilized data directly from input for the
    ``hist`` tree method. This DMatrix is primarily designed to save memory in training
    by avoiding intermediate storage. Set ``max_bin`` to control the number of bins
    during quantisation, which should be consistent with the training parameter
    ``max_bin``. When ``QuantileDMatrix`` is used for validation/test dataset, ``ref``
    should be another ``QuantileDMatrix``(or ``DMatrix``, but not recommended as it
    defeats the purpose of saving memory) constructed from training dataset.  See
    :py:obj:`xgboost.DMatrix` for documents on meta info.

    .. note::

        Do not use ``QuantileDMatrix`` as validation/test dataset without supplying a
        reference (the training dataset) ``QuantileDMatrix`` using ``ref`` as some
        information may be lost in quantisation.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    max_bin :
        The number of histogram bin, should be consistent with the training parameter
        ``max_bin``.

    ref :
        The training dataset that provides quantile information, needed when creating
        validation/test dataset with ``QuantileDMatrix``. Supplying the training DMatrix
        as a reference means that the same quantisation applied to the training data is
        applied to the validation/test data

    """

    @_deprecate_positional_args
    def __init__(self, data: DataType, label: Optional[ArrayLike]=None, *, weight: Optional[ArrayLike]=None, base_margin: Optional[ArrayLike]=None, missing: Optional[float]=None, silent: bool=False, feature_names: Optional[FeatureNames]=None, feature_types: Optional[FeatureTypes]=None, nthread: Optional[int]=None, max_bin: Optional[int]=None, ref: Optional[DMatrix]=None, group: Optional[ArrayLike]=None, qid: Optional[ArrayLike]=None, label_lower_bound: Optional[ArrayLike]=None, label_upper_bound: Optional[ArrayLike]=None, feature_weights: Optional[ArrayLike]=None, enable_categorical: bool=False, data_split_mode: DataSplitMode=DataSplitMode.ROW) -> None:
        self.max_bin = max_bin
        self.missing = missing if missing is not None else np.nan
        self.nthread = nthread if nthread is not None else -1
        self._silent = silent
        if isinstance(data, ctypes.c_void_p):
            self.handle = data
            return
        if qid is not None and group is not None:
            raise ValueError('Only one of the eval_qid or eval_group for each evaluation dataset should be provided.')
        if isinstance(data, DataIter):
            if any((info is not None for info in (label, weight, base_margin, feature_names, feature_types, group, qid, label_lower_bound, label_upper_bound, feature_weights))):
                raise ValueError('If data iterator is used as input, data like label should be specified as batch argument.')
        self._init(data, ref=ref, label=label, weight=weight, base_margin=base_margin, group=group, qid=qid, label_lower_bound=label_lower_bound, label_upper_bound=label_upper_bound, feature_weights=feature_weights, feature_names=feature_names, feature_types=feature_types, enable_categorical=enable_categorical)

    def _init(self, data: DataType, ref: Optional[DMatrix], enable_categorical: bool, **meta: Any) -> None:
        from .data import SingleBatchInternalIter, _is_dlpack, _is_iter, _transform_dlpack
        if _is_dlpack(data):
            data = _transform_dlpack(data)
        if _is_iter(data):
            it = data
        else:
            it = SingleBatchInternalIter(data=data, **meta)
        handle = ctypes.c_void_p()
        reset_callback, next_callback = it.get_callbacks(True, enable_categorical)
        if it.cache_prefix is not None:
            raise ValueError("QuantileDMatrix doesn't cache data, remove the cache_prefix in iterator to fix this error.")
        config = make_jcargs(nthread=self.nthread, missing=self.missing, max_bin=self.max_bin)
        ret = _LIB.XGQuantileDMatrixCreateFromCallback(None, it.proxy.handle, ref.handle if ref is not None else ref, reset_callback, next_callback, config, ctypes.byref(handle))
        it.reraise()
        _check_call(ret)
        self.handle = handle