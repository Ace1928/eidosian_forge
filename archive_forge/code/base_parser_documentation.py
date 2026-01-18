from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
import datetime
from enum import Enum
import itertools
from typing import (
import warnings
import numpy as np
from pandas._libs import (
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import (
from pandas.core import algorithms
from pandas.core.arrays import (
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools
from pandas.io.common import is_potential_multi_index

        Validate the 'usecols' parameter.

        Checks whether or not the 'usecols' parameter contains all integers
        (column selection by index), strings (column by name) or is a callable.
        Raises a ValueError if that is not the case.

        Parameters
        ----------
        usecols : list-like, callable, or None
            List of columns to use when parsing or a callable that can be used
            to filter a list of table columns.

        Returns
        -------
        usecols_tuple : tuple
            A tuple of (verified_usecols, usecols_dtype).

            'verified_usecols' is either a set if an array-like is passed in or
            'usecols' if a callable or None is passed in.

            'usecols_dtype` is the inferred dtype of 'usecols' if an array-like
            is passed in or None if a callable or None is passed in.
        