from typing import Any, Callable
import ibis
import pandas as pd
from ibis.backends.pandas import Backend
from triad.utils.assertion import assert_or_throw
from fugue import (
from fugue_ibis._utils import to_ibis_schema, to_schema
from .._compat import IbisTable
from .ibis_engine import IbisEngine, parse_ibis_engine
@parse_ibis_engine.candidate(lambda obj, *args, **kwargs: isinstance(obj, NativeExecutionEngine))
def _pd_to_ibis_engine(obj: Any, engine: ExecutionEngine) -> IbisEngine:
    return PandasIbisEngine(engine)