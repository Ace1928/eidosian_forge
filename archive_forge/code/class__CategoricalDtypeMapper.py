from __future__ import annotations
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
from pandas.core.dtypes.common import _get_dtype, is_string_dtype
from pyarrow.types import is_dictionary
from modin.pandas.indexing import is_range_like
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
class _CategoricalDtypeMapper:

    @staticmethod
    def __from_arrow__(arr):
        values = []
        categories = {}
        chunks = arr.chunks if isinstance(arr, pa.ChunkedArray) else (arr,)
        for chunk in chunks:
            assert isinstance(chunk, pa.DictionaryArray)
            cat = chunk.dictionary.to_pandas()
            values.append(chunk.indices.to_pandas().map(cat))
            categories.update(((c, None) for c in cat))
        return pandas.Categorical(pandas.concat(values, ignore_index=True), dtype=pandas.CategoricalDtype(categories, ordered=True))