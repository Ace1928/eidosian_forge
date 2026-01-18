from __future__ import annotations
import pickle
import warnings
from typing import TYPE_CHECKING, Union
import pandas
from pandas._typing import CompressionOptions, StorageOptions
from pandas.core.dtypes.dtypes import SparseDtype
from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.io import to_dask, to_ray
from modin.utils import _inherit_docstrings
class BaseSparseAccessor(ClassLogger):
    """
    Base class for various sparse DataFrame accessor classes.

    Parameters
    ----------
    data : DataFrame or Series
        Object to operate on.
    """
    _parent: Union[DataFrame, Series]
    _validation_msg = "Can only use the '.sparse' accessor with Sparse data."

    def __init__(self, data: Union[DataFrame, Series]=None):
        self._parent = data
        self._validate(data)

    @classmethod
    def _validate(cls, data: Union[DataFrame, Series]):
        """
        Verify that `data` dtypes are compatible with `pandas.core.dtypes.dtypes.SparseDtype`.

        Parameters
        ----------
        data : DataFrame or Series
            Object to check.

        Raises
        ------
        NotImplementedError
            Function is implemented in child classes.
        """
        raise NotImplementedError

    def _default_to_pandas(self, op, *args, **kwargs):
        """
        Convert dataset to pandas type and call a pandas sparse.`op` on it.

        Parameters
        ----------
        op : str
            Name of pandas function.
        *args : list
            Additional positional arguments to be passed in `op`.
        **kwargs : dict
            Additional keywords arguments to be passed in `op`.

        Returns
        -------
        object
            Result of operation.
        """
        return self._parent._default_to_pandas(lambda parent: op(parent.sparse, *args, **kwargs))