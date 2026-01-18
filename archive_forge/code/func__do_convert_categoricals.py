from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def _do_convert_categoricals(self, data: DataFrame, value_label_dict: dict[str, dict[float, str]], lbllist: Sequence[str], order_categoricals: bool) -> DataFrame:
    """
        Converts categorical columns to Categorical type.
        """
    if not value_label_dict:
        return data
    cat_converted_data = []
    for col, label in zip(data, lbllist):
        if label in value_label_dict:
            vl = value_label_dict[label]
            keys = np.array(list(vl.keys()))
            column = data[col]
            key_matches = column.isin(keys)
            if self._using_iterator and key_matches.all():
                initial_categories: np.ndarray | None = keys
            else:
                if self._using_iterator:
                    warnings.warn(categorical_conversion_warning, CategoricalConversionWarning, stacklevel=find_stack_level())
                initial_categories = None
            cat_data = Categorical(column, categories=initial_categories, ordered=order_categoricals)
            if initial_categories is None:
                categories = []
                for category in cat_data.categories:
                    if category in vl:
                        categories.append(vl[category])
                    else:
                        categories.append(category)
            else:
                categories = list(vl.values())
            try:
                cat_data = cat_data.rename_categories(categories)
            except ValueError as err:
                vc = Series(categories, copy=False).value_counts()
                repeated_cats = list(vc.index[vc > 1])
                repeats = '-' * 80 + '\n' + '\n'.join(repeated_cats)
                msg = f'\nValue labels for column {col} are not unique. These cannot be converted to\npandas categoricals.\n\nEither read the file with `convert_categoricals` set to False or use the\nlow level interface in `StataReader` to separately read the values and the\nvalue_labels.\n\nThe repeated labels are:\n{repeats}\n'
                raise ValueError(msg) from err
            cat_series = Series(cat_data, index=data.index, copy=False)
            cat_converted_data.append((col, cat_series))
        else:
            cat_converted_data.append((col, data[col]))
    data = DataFrame(dict(cat_converted_data), copy=False)
    return data