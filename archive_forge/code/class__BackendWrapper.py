from typing import Any, Callable
import ibis
import pandas as pd
from ibis.backends.pandas import Backend
from triad.utils.assertion import assert_or_throw
from fugue import (
from fugue_ibis._utils import to_ibis_schema, to_schema
from .._compat import IbisTable
from .ibis_engine import IbisEngine, parse_ibis_engine
class _BackendWrapper(Backend):

    def set_schemas(self, dfs: DataFrames) -> None:
        self._schemas = {k: to_ibis_schema(v.schema) for k, v in dfs.items()}

    def table(self, name: str, schema: Any=None):
        return super().table(name, schema=self._schemas[name] if schema is None and name in self._schemas else schema)