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
class DuckDBEngine(SQLEngine):
    """DuckDB SQL backend implementation.

    :param execution_engine: the execution engine this sql engine will run on
    """

    @property
    def dialect(self) -> Optional[str]:
        return 'duckdb'

    def select(self, dfs: DataFrames, statement: StructuredRawSQL) -> DataFrame:
        if isinstance(self.execution_engine, DuckExecutionEngine):
            return self._duck_select(dfs, statement)
        else:
            _dfs, _sql = self.encode(dfs, statement)
            return self._other_select(_dfs, _sql)

    def table_exists(self, table: str) -> bool:
        return self._get_table(table) is not None

    def save_table(self, df: DataFrame, table: str, mode: str='overwrite', partition_spec: Optional[PartitionSpec]=None, **kwargs: Any) -> None:
        if isinstance(self.execution_engine, DuckExecutionEngine):
            con = self.execution_engine.connection
            tdf: DuckDataFrame = _to_duck_df(self.execution_engine, df, create_view=False)
            et = self._get_table(table)
            if et is not None:
                if mode == 'overwrite':
                    tp = 'VIEW' if et['table_type'] == 'VIEW' else 'TABLE'
                    tn = encode_column_name(et['table_name'])
                    con.query(f'DROP {tp} {tn}')
                else:
                    raise Exception(f'{table} exists')
            tdf.native.create(table)
        else:
            raise NotImplementedError('save_table can only be used with DuckExecutionEngine')

    def load_table(self, table: str, **kwargs: Any) -> DataFrame:
        if isinstance(self.execution_engine, DuckExecutionEngine):
            return DuckDataFrame(self.execution_engine.connection.table(table))
        else:
            raise NotImplementedError('load_table can only be used with DuckExecutionEngine')

    @property
    def is_distributed(self) -> bool:
        return False

    def _duck_select(self, dfs: DataFrames, statement: StructuredRawSQL) -> DataFrame:
        name_map: Dict[str, str] = {}
        for k, v in dfs.items():
            tdf: DuckDataFrame = _to_duck_df(self.execution_engine, v, create_view=True)
            name_map[k] = tdf.alias
        query = statement.construct(name_map, dialect=self.dialect, log=self.log)
        result = self.execution_engine.connection.query(query)
        return DuckDataFrame(result)

    def _other_select(self, dfs: DataFrames, statement: str) -> DataFrame:
        conn = duckdb.connect()
        try:
            for k, v in dfs.items():
                duckdb.from_arrow(v.as_arrow(), connection=conn).create_view(k)
            return ArrowDataFrame(_duck_as_arrow(conn.execute(statement)))
        finally:
            conn.close()

    def _get_table(self, table: str) -> Optional[Dict[str, Any]]:
        if isinstance(self.execution_engine, DuckExecutionEngine):
            con = self.execution_engine.connection
            qt = quote_name(table, "'")
            if not qt.startswith("'"):
                qt = "'" + qt + "'"
            tables = con.query(f'SELECT table_catalog,table_schema,table_name,table_type FROM information_schema.tables WHERE table_name={qt}').to_df().to_dict('records')
            return None if len(tables) == 0 else tables[0]
        else:
            raise NotImplementedError('table_exists can only be used with DuckExecutionEngine')