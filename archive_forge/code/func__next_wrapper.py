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
def _next_wrapper(self, this: None) -> int:
    """A wrapper for user defined `next` function.

        `this` is not used in Python.  ctypes can handle `self` of a Python
        member function automatically when converting it to c function
        pointer.

        """

    @require_keyword_args(True)
    def input_data(*, data: Any, feature_names: Optional[FeatureNames]=None, feature_types: Optional[FeatureTypes]=None, **kwargs: Any) -> None:
        from .data import _proxy_transform, dispatch_proxy_set_data
        try:
            ref = weakref.ref(data)
        except TypeError:
            ref = None
        if self._temporary_data is not None and ref is not None and (ref is self._data_ref):
            new, cat_codes, feature_names, feature_types = self._temporary_data
        else:
            new, cat_codes, feature_names, feature_types = _proxy_transform(data, feature_names, feature_types, self._enable_categorical)
        self._temporary_data = (new, cat_codes, feature_names, feature_types)
        dispatch_proxy_set_data(self.proxy, new, cat_codes, self._allow_host)
        self.proxy.set_info(feature_names=feature_names, feature_types=feature_types, **kwargs)
        self._data_ref = ref
    return self._handle_exception(lambda: self.next(input_data), 0)