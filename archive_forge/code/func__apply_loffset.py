from __future__ import annotations
import copy
import datetime
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core import dtypes, duck_array_ops, nputils, ops
from xarray.core._aggregations import (
from xarray.core.alignment import align
from xarray.core.arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.concat import concat
from xarray.core.formatting import format_array_flat
from xarray.core.indexes import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import IndexVariable, Variable
from xarray.util.deprecation_helpers import _deprecate_positional_args
def _apply_loffset(loffset: str | pd.DateOffset | datetime.timedelta | pd.Timedelta, result: pd.Series | pd.DataFrame):
    """
    (copied from pandas)
    if loffset is set, offset the result index

    This is NOT an idempotent routine, it will be applied
    exactly once to the result.

    Parameters
    ----------
    result : Series or DataFrame
        the result of resample
    """
    if not isinstance(loffset, (str, pd.DateOffset, datetime.timedelta)):
        raise ValueError(f'`loffset` must be a str, pd.DateOffset, datetime.timedelta, or pandas.Timedelta object. Got {loffset}.')
    if isinstance(loffset, str):
        loffset = pd.tseries.frequencies.to_offset(loffset)
    needs_offset = isinstance(loffset, (pd.DateOffset, datetime.timedelta)) and isinstance(result.index, pd.DatetimeIndex) and (len(result.index) > 0)
    if needs_offset:
        result.index = result.index + loffset