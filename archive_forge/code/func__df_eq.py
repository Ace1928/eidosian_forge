import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from fsspec import AbstractFileSystem
from triad import Schema, assert_or_throw
from triad.collections.schema import SchemaError
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.io import url_to_fs
from triad.utils.pyarrow import pa_batch_to_dicts
from .api import as_fugue_df, get_column_names, normalize_column_names, rename
from .dataframe import DataFrame, LocalBoundedDataFrame
def _df_eq(df: DataFrame, data: Any, schema: Any=None, digits=8, check_order: bool=False, check_schema: bool=True, check_content: bool=True, no_pandas: bool=False, throw=False) -> bool:
    """Compare if two dataframes are equal. Is for internal, unit test
    purpose only. It will convert both dataframes to
    :class:`~fugue.dataframe.dataframe.LocalBoundedDataFrame`, so it assumes
    both dataframes are small and fast enough to convert. DO NOT use it
    on critical or expensive tasks.

    :param df: first data frame
    :param data: :ref:`DataFrame like
      <tutorial:tutorials/advanced/x-like:dataframe>` object
    :param schema: :ref:`Schema like
      <tutorial:tutorials/advanced/x-like:schema>` object, defaults to None
    :param digits: precision on float number comparison, defaults to 8
    :param check_order: if to compare the row orders, defaults to False
    :param check_schema: if compare schemas, defaults to True
    :param check_content: if to compare the row values, defaults to True
    :param no_pandas: if true, it will compare the string representations of the
      dataframes, otherwise, it will convert both to pandas dataframe to compare,
      defaults to False
    :param throw: if to throw error if not equal, defaults to False
    :return: if they equal
    """
    df1 = as_fugue_df(df).as_local_bounded()
    if schema is not None:
        df2 = as_fugue_df(data, schema=schema).as_local_bounded()
    else:
        df2 = as_fugue_df(data).as_local_bounded()
    try:
        assert df1.count() == df2.count(), f'count mismatch {df1.count()}, {df2.count()}'
        assert not check_schema or _schema_eq(df.schema, df2.schema), f'schema mismatch {df.schema.pa_schema}, {df2.schema.pa_schema}'
        if not check_content:
            return True
        cols: Any = df1.columns
        if no_pandas:
            dd1 = [[x.__repr__()] for x in df1.as_array_iterable(type_safe=True)]
            dd2 = [[x.__repr__()] for x in df2.as_array_iterable(type_safe=True)]
            d1 = pd.DataFrame(dd1, columns=['data'])
            d2 = pd.DataFrame(dd2, columns=['data'])
            cols = ['data']
        else:
            d1 = df1.as_pandas()
            d2 = df2.as_pandas()
        if not check_order:
            d1 = d1.sort_values(cols)
            d2 = d2.sort_values(cols)
        d1 = d1.reset_index(drop=True)
        d2 = d2.reset_index(drop=True)
        pd.testing.assert_frame_equal(d1, d2, rtol=0, atol=10 ** (-digits), check_dtype=False, check_exact=False)
        return True
    except AssertionError:
        if throw:
            raise
        return False