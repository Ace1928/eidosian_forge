from typing import Any, Callable, Dict
import ibis
from fugue import DataFrame, DataFrames, Processor, WorkflowDataFrame
from fugue.exceptions import FugueWorkflowCompileError
from fugue.workflow.workflow import WorkflowDataFrames
from triad import assert_or_throw, extension_method
from ._utils import LazyIbisObject, _materialize
from .execution.ibis_engine import parse_ibis_engine
from ._compat import IbisTable
class _IbisProcessor(Processor):

    def process(self, dfs: DataFrames) -> DataFrame:
        ibis_func = self.params.get_or_throw('ibis_func', Callable)
        ibis_engine = self.params.get_or_none('ibis_engine', object)
        ie = parse_ibis_engine(self.execution_engine if ibis_engine is None else ibis_engine, self.execution_engine)
        return ie.select(dfs, ibis_func)