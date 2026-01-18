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
def _duck_select(self, dfs: DataFrames, statement: StructuredRawSQL) -> DataFrame:
    name_map: Dict[str, str] = {}
    for k, v in dfs.items():
        tdf: DuckDataFrame = _to_duck_df(self.execution_engine, v, create_view=True)
        name_map[k] = tdf.alias
    query = statement.construct(name_map, dialect=self.dialect, log=self.log)
    result = self.execution_engine.connection.query(query)
    return DuckDataFrame(result)