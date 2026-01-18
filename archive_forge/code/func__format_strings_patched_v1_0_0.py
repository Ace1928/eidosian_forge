import numbers
import os
from packaging.version import Version
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._typing import Dtype
from pandas.compat import set_function_name
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.indexers import check_array_indexer, validate_indices
from pandas.io.formats.format import ExtensionArrayFormatter
from ray.air.util.tensor_extensions.utils import (
from ray.util.annotations import PublicAPI
def _format_strings_patched_v1_0_0(self) -> List[str]:
    from functools import partial
    from pandas.core.construction import extract_array
    from pandas.io.formats.format import format_array
    from pandas.io.formats.printing import pprint_thing
    if not isinstance(self.values, TensorArray):
        return self._format_strings_orig()
    values = extract_array(self.values, extract_numpy=True)
    array = np.asarray(values)
    if array.ndim == 1:
        return self._format_strings_orig()

    def format_array_wrap(array_, formatter_):
        fmt_values = format_array(array_, formatter_, float_format=self.float_format, na_rep=self.na_rep, digits=self.digits, space=self.space, justify=self.justify, decimal=self.decimal, leading_space=self.leading_space)
        return fmt_values
    flat_formatter = self.formatter
    if flat_formatter is None:
        flat_formatter = values._formatter(boxed=True)
    flat_array = array.ravel('K')
    fmt_flat_array = np.asarray(format_array_wrap(flat_array, flat_formatter))
    order = 'F' if array.flags.f_contiguous else 'C'
    fmt_array = fmt_flat_array.reshape(array.shape, order=order)

    def format_strings_slim(array_, leading_space):
        formatter = partial(pprint_thing, escape_chars=('\t', '\r', '\n'))

        def _format(x):
            return str(formatter(x))
        fmt_values = []
        for v in array_:
            tpl = '{v}' if leading_space is False else ' {v}'
            fmt_values.append(tpl.format(v=_format(v)))
        return fmt_values
    return format_strings_slim(fmt_array, self.leading_space)