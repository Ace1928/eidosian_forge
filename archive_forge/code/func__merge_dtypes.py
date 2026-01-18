from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
@classmethod
def _merge_dtypes(cls, values: list[Union['DtypesDescriptor', pandas.Series, None]]) -> 'DtypesDescriptor':
    """
        Union columns described by ``values`` and compute common dtypes for them.

        Parameters
        ----------
        values : list of DtypesDescriptors, pandas.Series or Nones

        Returns
        -------
        DtypesDescriptor
        """
    known_dtypes = {}
    cols_with_unknown_dtypes = []
    know_all_names = True
    dtypes_are_unknown = False
    dtypes_matrix = pandas.DataFrame()
    for i, val in enumerate(values):
        if isinstance(val, cls):
            know_all_names &= val._know_all_names
            dtypes = val._known_dtypes.copy()
            dtypes.update({col: 'unknown' for col in val._cols_with_unknown_dtypes})
            if val._remaining_dtype is not None:
                know_all_names = False
            series = pandas.Series(dtypes, name=i)
            dtypes_matrix = pandas.concat([dtypes_matrix, series], axis=1)
            dtypes_matrix.fillna(value={i: pandas.api.types.pandas_dtype(float) if val._know_all_names and val._remaining_dtype is None else 'unknown'}, inplace=True)
        elif isinstance(val, pandas.Series):
            dtypes_matrix = pandas.concat([dtypes_matrix, val], axis=1)
        elif val is None:
            dtypes_are_unknown = True
            know_all_names = False
        else:
            raise NotImplementedError(type(val))
    if dtypes_are_unknown:
        return DtypesDescriptor(cols_with_unknown_dtypes=dtypes_matrix.index.tolist(), know_all_names=know_all_names)

    def combine_dtypes(row):
        if (row == 'unknown').any():
            return 'unknown'
        row = row.fillna(pandas.api.types.pandas_dtype('float'))
        return find_common_type(list(row.values))
    dtypes = dtypes_matrix.apply(combine_dtypes, axis=1)
    for col, dtype in dtypes.items():
        if dtype == 'unknown':
            cols_with_unknown_dtypes.append(col)
        else:
            known_dtypes[col] = dtype
    return DtypesDescriptor(known_dtypes, cols_with_unknown_dtypes, remaining_dtype=None, know_all_names=know_all_names)