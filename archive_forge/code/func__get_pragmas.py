import logging
from typing import Any, Dict, Iterable, List, Optional, Union
import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from triad import SerializableRLock
from triad.utils.assertion import assert_or_throw
from triad.utils.schema import quote_name
from fugue import (
from fugue.collections.partition import PartitionSpec, parse_presort_exp
from fugue.collections.sql import StructuredRawSQL, TempTableName
from fugue.dataframe import DataFrame, DataFrames, LocalBoundedDataFrame
from fugue.dataframe.utils import get_join_schemas
from ._io import DuckDBIO
from ._utils import (
from .dataframe import DuckDataFrame, _duck_as_arrow
def _get_pragmas(self) -> Iterable[str]:
    for k, v in self.conf.items():
        if k.startswith(_FUGUE_DUCKDB_PRAGMA_CONFIG_PREFIX):
            name = k[len(_FUGUE_DUCKDB_PRAGMA_CONFIG_PREFIX):]
            assert_or_throw(name.isidentifier(), ValueError(f'{name} is not a valid pragma key'))
            value = encode_value_to_expr(v)
            yield f'PRAGMA {name}={value};'