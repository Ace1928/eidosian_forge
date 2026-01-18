from inspect import signature
from math import prod
import numpy
import pandas
from pandas.api.types import is_scalar
from pandas.core.dtypes.common import is_bool_dtype, is_list_like, is_numeric_dtype
import modin.pandas as pd
from modin.core.dataframe.algebra import Binary, Map, Reduce
from modin.error_message import ErrorMessage
from .utils import try_convert_from_interoperable_type
def _build_repr_array(self):

    def _generate_indices_for_axis(axis_size, num_elements=numpy.get_printoptions()['edgeitems']):
        if axis_size > num_elements * 2:
            return list(range(num_elements + 1)) + list(range(axis_size - num_elements, axis_size))
        return list(range(axis_size))
    if self._ndim == 1 or self.shape[1] == 0:
        idxs = _generate_indices_for_axis(len(self))
        arr = self._query_compiler.getitem_row_array(idxs).to_numpy()
        if self._ndim == 1:
            arr = arr.flatten()
    elif self.shape[0] == 1:
        idxs = _generate_indices_for_axis(self.shape[1])
        arr = self._query_compiler.getitem_column_array(idxs).to_numpy()
    else:
        row_idxs = _generate_indices_for_axis(len(self))
        col_idxs = _generate_indices_for_axis(self.shape[1])
        arr = self._query_compiler.take_2d_positional(row_idxs, col_idxs).to_numpy()
    return arr