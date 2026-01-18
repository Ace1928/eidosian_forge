from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
def _case_when_caselists():

    def permutations(values):
        return [p for r in range(1, len(values) + 1) for p in itertools.permutations(values, r)]
    conditions = permutations([[True, False, False, False] * 10, pandas.Series([True, False, False, False] * 10), pandas.Series([True, False, False, False] * 10, index=range(78, -2, -2)), lambda df: df.gt(0)])
    replacements = permutations([[0, 3, 4, 5] * 10, 0, lambda df: 1])
    caselists = []
    for c in conditions:
        for r in replacements:
            if len(c) == len(r):
                caselists.append(list(zip(c, r)))
    return caselists