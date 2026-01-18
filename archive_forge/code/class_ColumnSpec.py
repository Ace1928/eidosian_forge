from __future__ import annotations
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_220, PANDAS_GE_300
from dask.dataframe._pyarrow import is_object_string_dtype
from dask.dataframe.core import tokenize
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.utils import random_state_data
@dataclass
class ColumnSpec:
    """Encapsulates properties of a family of columns with the same dtype.
    Different method can be specified for integer dtype ("poisson", "uniform",
    "binomial", etc.)

    Notes
    -----
    This API is still experimental, and will likely change in the future"""
    prefix: str | None = None
    'Column prefix. If not specified, will default to str(dtype)'
    dtype: str | type | None = None
    'Column data type. Only supports numpy dtypes'
    number: int = 1
    'How many columns to create with these properties. Default 1.\n    If more than one columns are specified, they will be numbered: "int1", "int2", etc.'
    nunique: int | None = None
    'For a "category" column, how many unique categories to generate'
    choices: list = field(default_factory=list)
    'For a "category" or str column, list of possible values'
    low: int | None = None
    "Start value for an int column. Optional if random=True, since ``randint`` doesn't accept\n    high and low."
    high: int | None = None
    'For an int column, high end of range'
    length: int | None = None
    'For a str or "category" column with random=True, how large a string to generate'
    random: bool = False
    'For an int column, whether to use ``randint``. For a string column produces a random string\n    of specified ``length``'
    method: str | None = None
    'For an int column, method to use when generating the value, such as "poisson", "uniform", "binomial".\n    Default "poisson". Delegates to the same method of ``RandomState``'
    args: tuple[Any, ...] = field(default_factory=tuple)
    'Args to pass into the method'
    kwargs: dict[str, Any] = field(default_factory=dict)
    'Any other kwargs to pass into the method'