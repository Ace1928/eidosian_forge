from typing import Any, Callable, Dict, Optional, Tuple
import ibis
from ibis.backends.pandas import Backend
from fugue import DataFrame, DataFrames, ExecutionEngine
from fugue.collections.sql import StructuredRawSQL, TempTableName
from fugue_ibis import IbisTable
from fugue_ibis._utils import to_ibis_schema
from fugue_ibis.execution.ibis_engine import IbisEngine, parse_ibis_engine
from .execution_engine import DuckDBEngine, DuckExecutionEngine
@parse_ibis_engine.candidate(lambda obj, *args, **kwargs: isinstance(obj, DuckExecutionEngine) or (isinstance(obj, str) and obj in ['duck', 'duckdb']))
def _to_duck_ibis_engine(obj: Any, engine: ExecutionEngine) -> Optional[IbisEngine]:
    return DuckDBIbisEngine(engine)