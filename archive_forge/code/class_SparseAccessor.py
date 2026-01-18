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
@_inherit_docstrings(pandas.core.arrays.sparse.accessor.SparseAccessor)
class SparseAccessor(BaseSparseAccessor):

    @classmethod
    def _validate(cls, data: Series):
        """
        Verify that `data` dtype is compatible with `pandas.core.dtypes.dtypes.SparseDtype`.

        Parameters
        ----------
        data : Series
            Object to check.

        Raises
        ------
        AttributeError
            If check fails.
        """
        if not isinstance(data.dtype, SparseDtype):
            raise AttributeError(cls._validation_msg)

    @property
    def density(self):
        return self._parent._default_to_pandas(pandas.Series.sparse).density

    @property
    def fill_value(self):
        return self._parent._default_to_pandas(pandas.Series.sparse).fill_value

    @property
    def npoints(self):
        return self._parent._default_to_pandas(pandas.Series.sparse).npoints

    @property
    def sp_values(self):
        return self._parent._default_to_pandas(pandas.Series.sparse).sp_values

    @classmethod
    def from_coo(cls, A, dense_index=False):
        return cls._default_to_pandas(pandas.Series.sparse.from_coo, A, dense_index=dense_index)

    def to_coo(self, row_levels=(0,), column_levels=(1,), sort_labels=False):
        return self._default_to_pandas(pandas.Series.sparse.to_coo, row_levels=row_levels, column_levels=column_levels, sort_labels=sort_labels)

    def to_dense(self):
        return self._default_to_pandas(pandas.Series.sparse.to_dense)