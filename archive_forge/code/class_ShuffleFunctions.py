import abc
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._libs.tslibs import to_offset
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype
from pandas.core.resample import _get_timestamp_range_edges
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
class ShuffleFunctions:
    """
    Defines an interface to perform the sampling, quantiles picking, and the splitting stages for the range-partitioning building.

    Parameters
    ----------
    modin_frame : PandasDataframe
        The frame to build the range-partitioning for.
    columns : str or list of strings
        The column/columns to use as a key.
    ascending : bool
        Whether the ranges should be in ascending or descending order.
    ideal_num_new_partitions : int
        The ideal number of new partitions.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, modin_frame, columns, ascending, ideal_num_new_partitions, **kwargs):
        pass

    @abc.abstractmethod
    def sample_fn(self, partition: pandas.DataFrame) -> pandas.DataFrame:
        """
        Pick samples over the given partition.

        Parameters
        ----------
        partition : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame:
            The samples for the partition.
        """
        pass

    @abc.abstractmethod
    def pivot_fn(self, samples: 'list[pandas.DataFrame]') -> int:
        """
        Determine quantiles from the given samples and save it for the future ``.split_fn()`` calls.

        Parameters
        ----------
        samples : list of pandas.DataFrames

        Returns
        -------
        int
            The number of bins the ``.split_fn()`` will return.
        """
        pass

    @abc.abstractmethod
    def split_fn(self, partition: pandas.DataFrame) -> 'tuple[pandas.DataFrame, ...]':
        """
        Split the given dataframe into the range-partitions defined by the preceding call of the ``.pivot_fn()``.

        Parameters
        ----------
        partition : pandas.DataFrame

        Returns
        -------
        tuple of pandas.DataFrames

        Notes
        -----
        In order to call this method you must call the ``.pivot_fn()`` first.
        """
        pass