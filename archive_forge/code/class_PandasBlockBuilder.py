import collections
import heapq
from typing import (
import numpy as np
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data._internal.util import find_partitions
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.row import TableRow
class PandasBlockBuilder(TableBlockBuilder):

    def __init__(self):
        pandas = lazy_import_pandas()
        super().__init__(pandas.DataFrame)

    @staticmethod
    def _table_from_pydict(columns: Dict[str, List[Any]]) -> 'pandas.DataFrame':
        pandas = lazy_import_pandas()
        for key, value in columns.items():
            if key == TENSOR_COLUMN_NAME or isinstance(next(iter(value), None), np.ndarray):
                from ray.data.extensions.tensor_extension import TensorArray
                columns[key] = TensorArray(value)
        return pandas.DataFrame(columns)

    @staticmethod
    def _concat_tables(tables: List['pandas.DataFrame']) -> 'pandas.DataFrame':
        pandas = lazy_import_pandas()
        from ray.air.util.data_batch_conversion import _cast_ndarray_columns_to_tensor_extension
        if len(tables) > 1:
            df = pandas.concat(tables, ignore_index=True)
            df.reset_index(drop=True, inplace=True)
        else:
            df = tables[0]
        ctx = DataContext.get_current()
        if ctx.enable_tensor_extension_casting:
            df = _cast_ndarray_columns_to_tensor_extension(df)
        return df

    @staticmethod
    def _concat_would_copy() -> bool:
        return True

    @staticmethod
    def _empty_table() -> 'pandas.DataFrame':
        pandas = lazy_import_pandas()
        return pandas.DataFrame()