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
def _infer_predict_output(booster: Booster, features: int, is_df: bool, inplace: bool, **kwargs: Any) -> Tuple[Tuple[int, ...], Dict[int, str]]:
    """Create a dummy test sample to infer output shape for prediction."""
    assert isinstance(features, int)
    rng = numpy.random.RandomState(1994)
    test_sample = rng.randn(1, features)
    if inplace:
        kwargs = kwargs.copy()
        if kwargs.pop('predict_type') == 'margin':
            kwargs['output_margin'] = True
    m = DMatrix(test_sample, enable_categorical=True)
    test_predt = booster.predict(m, validate_features=False, **kwargs)
    n_columns = test_predt.shape[1] if len(test_predt.shape) > 1 else 1
    meta: Dict[int, str] = {}
    if _can_output_df(is_df, test_predt.shape):
        for i in range(n_columns):
            meta[i] = 'f4'
    return (test_predt.shape, meta)