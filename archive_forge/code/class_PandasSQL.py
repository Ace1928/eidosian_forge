from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
class PandasSQL(PandasObject, ABC):
    """
    Subclasses Should define read_query and to_sql.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        pass

    def read_table(self, table_name: str, index_col: str | list[str] | None=None, coerce_float: bool=True, parse_dates=None, columns=None, schema: str | None=None, chunksize: int | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> DataFrame | Iterator[DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def read_query(self, sql: str, index_col: str | list[str] | None=None, coerce_float: bool=True, parse_dates=None, params=None, chunksize: int | None=None, dtype: DtypeArg | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> DataFrame | Iterator[DataFrame]:
        pass

    @abstractmethod
    def to_sql(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append']='fail', index: bool=True, index_label=None, schema=None, chunksize: int | None=None, dtype: DtypeArg | None=None, method: Literal['multi'] | Callable | None=None, engine: str='auto', **engine_kwargs) -> int | None:
        pass

    @abstractmethod
    def execute(self, sql: str | Select | TextClause, params=None):
        pass

    @abstractmethod
    def has_table(self, name: str, schema: str | None=None) -> bool:
        pass

    @abstractmethod
    def _create_sql_schema(self, frame: DataFrame, table_name: str, keys: list[str] | None=None, dtype: DtypeArg | None=None, schema: str | None=None) -> str:
        pass