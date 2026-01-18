from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, union_categoricals
from dask.array.core import Array
from dask.array.dispatch import percentile_lookup
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
from dask.dataframe._compat import PANDAS_GE_220, is_any_real_numeric_dtype
from dask.dataframe.core import DataFrame, Index, Scalar, Series, _Frame
from dask.dataframe.dispatch import (
from dask.dataframe.extensions import make_array_nonempty, make_scalar
from dask.dataframe.utils import (
from dask.sizeof import SimpleSizeof, sizeof
from dask.utils import is_arraylike, is_series_like, typename
class DataFrameBackendEntrypoint(DaskBackendEntrypoint):
    """Dask-DataFrame version of ``DaskBackendEntrypoint``

    See Also
    --------
    PandasBackendEntrypoint
    """

    @staticmethod
    def from_dict(data: dict, *, npartitions: int, **kwargs):
        """Create a DataFrame collection from a dictionary

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.
        npartitions : int
            The desired number of output partitions.
        **kwargs :
            Optional backend kwargs.

        See Also
        --------
        dask.dataframe.io.io.from_dict
        """
        raise NotImplementedError

    @staticmethod
    def read_parquet(path: str | list, **kwargs):
        """Read Parquet files into a DataFrame collection

        Parameters
        ----------
        path : str or list
            Source path(s).
        **kwargs :
            Optional backend kwargs.

        See Also
        --------
        dask.dataframe.io.parquet.core.read_parquet
        """
        raise NotImplementedError

    @staticmethod
    def read_json(url_path: str | list, **kwargs):
        """Read json files into a DataFrame collection

        Parameters
        ----------
        url_path : str or list
            Source path(s).
        **kwargs :
            Optional backend kwargs.

        See Also
        --------
        dask.dataframe.io.json.read_json
        """
        raise NotImplementedError

    @staticmethod
    def read_orc(path: str | list, **kwargs):
        """Read ORC files into a DataFrame collection

        Parameters
        ----------
        path : str or list
            Source path(s).
        **kwargs :
            Optional backend kwargs.

        See Also
        --------
        dask.dataframe.io.orc.core.read_orc
        """
        raise NotImplementedError

    @staticmethod
    def read_csv(urlpath: str | list, **kwargs):
        """Read CSV files into a DataFrame collection

        Parameters
        ----------
        urlpath : str or list
            Source path(s).
        **kwargs :
            Optional backend kwargs.

        See Also
        --------
        dask.dataframe.io.csv.read_csv
        """
        raise NotImplementedError

    @staticmethod
    def read_hdf(pattern: str | list, key: str, **kwargs):
        """Read HDF5 files into a DataFrame collection

        Parameters
        ----------
        pattern : str or list
            Source path(s).
        key : str
            Group identifier in the store.
        **kwargs :
            Optional backend kwargs.

        See Also
        --------
        dask.dataframe.io.hdf.read_hdf
        """
        raise NotImplementedError