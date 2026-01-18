import atexit
import contextlib
import functools
import importlib
import inspect
import os
import os.path as op
import re
import shutil
import sys
import tempfile
import unittest
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from subprocess import STDOUT, CalledProcessError, TimeoutExpired, check_output
from unittest import TestCase
import joblib
import numpy as np
import scipy as sp
from numpy.testing import assert_allclose as np_assert_allclose
from numpy.testing import (
import sklearn
from sklearn.utils import (
from sklearn.utils._array_api import _check_array_api_dispatch
from sklearn.utils.fixes import VisibleDeprecationWarning, parse_version, sp_version
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
def _convert_container(container, constructor_name, columns_name=None, dtype=None, minversion=None, categorical_feature_names=None):
    """Convert a given container to a specific array-like with a dtype.

    Parameters
    ----------
    container : array-like
        The container to convert.
    constructor_name : {"list", "tuple", "array", "sparse", "dataframe",             "series", "index", "slice", "sparse_csr", "sparse_csc"}
        The type of the returned container.
    columns_name : index or array-like, default=None
        For pandas container supporting `columns_names`, it will affect
        specific names.
    dtype : dtype, default=None
        Force the dtype of the container. Does not apply to `"slice"`
        container.
    minversion : str, default=None
        Minimum version for package to install.
    categorical_feature_names : list of str, default=None
        List of column names to cast to categorical dtype.

    Returns
    -------
    converted_container
    """
    if constructor_name == 'list':
        if dtype is None:
            return list(container)
        else:
            return np.asarray(container, dtype=dtype).tolist()
    elif constructor_name == 'tuple':
        if dtype is None:
            return tuple(container)
        else:
            return tuple(np.asarray(container, dtype=dtype).tolist())
    elif constructor_name == 'array':
        return np.asarray(container, dtype=dtype)
    elif constructor_name in ('pandas', 'dataframe'):
        pd = pytest.importorskip('pandas', minversion=minversion)
        result = pd.DataFrame(container, columns=columns_name, dtype=dtype, copy=False)
        if categorical_feature_names is not None:
            for col_name in categorical_feature_names:
                result[col_name] = result[col_name].astype('category')
        return result
    elif constructor_name == 'pyarrow':
        pa = pytest.importorskip('pyarrow', minversion=minversion)
        array = np.asarray(container)
        if columns_name is None:
            columns_name = [f'col{i}' for i in range(array.shape[1])]
        data = {name: array[:, i] for i, name in enumerate(columns_name)}
        result = pa.Table.from_pydict(data)
        if categorical_feature_names is not None:
            for col_idx, col_name in enumerate(result.column_names):
                if col_name in categorical_feature_names:
                    result = result.set_column(col_idx, col_name, result.column(col_name).dictionary_encode())
        return result
    elif constructor_name == 'polars':
        pl = pytest.importorskip('polars', minversion=minversion)
        result = pl.DataFrame(container, schema=columns_name, orient='row')
        if categorical_feature_names is not None:
            for col_name in categorical_feature_names:
                result = result.with_columns(pl.col(col_name).cast(pl.Categorical))
        return result
    elif constructor_name == 'series':
        pd = pytest.importorskip('pandas', minversion=minversion)
        return pd.Series(container, dtype=dtype)
    elif constructor_name == 'index':
        pd = pytest.importorskip('pandas', minversion=minversion)
        return pd.Index(container, dtype=dtype)
    elif constructor_name == 'slice':
        return slice(container[0], container[1])
    elif 'sparse' in constructor_name:
        if not sp.sparse.issparse(container):
            container = np.atleast_2d(container)
        if 'array' in constructor_name and sp_version < parse_version('1.8'):
            raise ValueError(f'{constructor_name} is only available with scipy>=1.8.0, got {sp_version}')
        if constructor_name in ('sparse', 'sparse_csr'):
            return sp.sparse.csr_matrix(container, dtype=dtype)
        elif constructor_name == 'sparse_csr_array':
            return sp.sparse.csr_array(container, dtype=dtype)
        elif constructor_name == 'sparse_csc':
            return sp.sparse.csc_matrix(container, dtype=dtype)
        elif constructor_name == 'sparse_csc_array':
            return sp.sparse.csc_array(container, dtype=dtype)