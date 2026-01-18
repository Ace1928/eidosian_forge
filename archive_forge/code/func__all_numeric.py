from __future__ import annotations
import collections
import itertools as it
import operator
import uuid
import warnings
from functools import partial, wraps
from numbers import Integral
import numpy as np
import pandas as pd
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.dataframe._compat import (
from dask.dataframe.core import (
from dask.dataframe.dispatch import grouper_dispatch
from dask.dataframe.methods import concat, drop_columns
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import (
def _all_numeric(self):
    """Are all columns that we're not grouping on numeric?"""
    numerics = self.obj._meta._get_numeric_data()
    post_group_columns = self._meta.count().columns
    return len(set(post_group_columns) - set(numerics.columns)) == 0