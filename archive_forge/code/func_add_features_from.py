import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def add_features_from(self, other: 'Dataset') -> 'Dataset':
    """Add features from other Dataset to the current Dataset.

        Both Datasets must be constructed before calling this method.

        Parameters
        ----------
        other : Dataset
            The Dataset to take features from.

        Returns
        -------
        self : Dataset
            Dataset with the new features added.
        """
    if self._handle is None or other._handle is None:
        raise ValueError('Both source and target Datasets must be constructed before adding features')
    _safe_call(_LIB.LGBM_DatasetAddFeaturesFrom(self._handle, other._handle))
    was_none = self.data is None
    old_self_data_type = type(self.data).__name__
    if other.data is None:
        self.data = None
    elif self.data is not None:
        if isinstance(self.data, np.ndarray):
            if isinstance(other.data, np.ndarray):
                self.data = np.hstack((self.data, other.data))
            elif isinstance(other.data, scipy.sparse.spmatrix):
                self.data = np.hstack((self.data, other.data.toarray()))
            elif isinstance(other.data, pd_DataFrame):
                self.data = np.hstack((self.data, other.data.values))
            elif isinstance(other.data, dt_DataTable):
                self.data = np.hstack((self.data, other.data.to_numpy()))
            else:
                self.data = None
        elif isinstance(self.data, scipy.sparse.spmatrix):
            sparse_format = self.data.getformat()
            if isinstance(other.data, np.ndarray) or isinstance(other.data, scipy.sparse.spmatrix):
                self.data = scipy.sparse.hstack((self.data, other.data), format=sparse_format)
            elif isinstance(other.data, pd_DataFrame):
                self.data = scipy.sparse.hstack((self.data, other.data.values), format=sparse_format)
            elif isinstance(other.data, dt_DataTable):
                self.data = scipy.sparse.hstack((self.data, other.data.to_numpy()), format=sparse_format)
            else:
                self.data = None
        elif isinstance(self.data, pd_DataFrame):
            if not PANDAS_INSTALLED:
                raise LightGBMError('Cannot add features to DataFrame type of raw data without pandas installed. Install pandas and restart your session.')
            if isinstance(other.data, np.ndarray):
                self.data = concat((self.data, pd_DataFrame(other.data)), axis=1, ignore_index=True)
            elif isinstance(other.data, scipy.sparse.spmatrix):
                self.data = concat((self.data, pd_DataFrame(other.data.toarray())), axis=1, ignore_index=True)
            elif isinstance(other.data, pd_DataFrame):
                self.data = concat((self.data, other.data), axis=1, ignore_index=True)
            elif isinstance(other.data, dt_DataTable):
                self.data = concat((self.data, pd_DataFrame(other.data.to_numpy())), axis=1, ignore_index=True)
            else:
                self.data = None
        elif isinstance(self.data, dt_DataTable):
            if isinstance(other.data, np.ndarray):
                self.data = dt_DataTable(np.hstack((self.data.to_numpy(), other.data)))
            elif isinstance(other.data, scipy.sparse.spmatrix):
                self.data = dt_DataTable(np.hstack((self.data.to_numpy(), other.data.toarray())))
            elif isinstance(other.data, pd_DataFrame):
                self.data = dt_DataTable(np.hstack((self.data.to_numpy(), other.data.values)))
            elif isinstance(other.data, dt_DataTable):
                self.data = dt_DataTable(np.hstack((self.data.to_numpy(), other.data.to_numpy())))
            else:
                self.data = None
        else:
            self.data = None
    if self.data is None:
        err_msg = f'Cannot add features from {type(other.data).__name__} type of raw data to {old_self_data_type} type of raw data.\n'
        err_msg += 'Set free_raw_data=False when construct Dataset to avoid this' if was_none else 'Freeing raw data'
        _log_warning(err_msg)
    self.feature_name = self.get_feature_name()
    _log_warning('Reseting categorical features.\nYou can set new categorical features via ``set_categorical_feature`` method')
    self.categorical_feature = 'auto'
    self.pandas_categorical = None
    return self