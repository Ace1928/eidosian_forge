from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
class QPDWorkflow(object):

    def __init__(self, ctx: QPDWorkflowContext):
        self._spec = WorkflowSpec()
        self._tasks: Dict[str, QPDTask] = {}
        self._ctx = ctx
        self._dfs = DataFrames({k: self._extract_df(k) for k in ctx.dfs.keys()})

    @property
    def ctx(self) -> QPDWorkflowContext:
        return self._ctx

    @property
    def dfs(self) -> DataFrames:
        return self._dfs

    def run(self) -> Any:
        self.ctx.run(self._spec, {})
        return self.ctx.result

    def const_to_col(self, value: Any, name: str='') -> WorkflowColumn:
        value = CreateConstColumn(value, name)
        value = self._add_task(value)
        return WorkflowColumn(self, value, name)

    def op_to_col(self, op: str, *args: Any, **kwargs: Any) -> WorkflowColumn:
        task = self.add(op, *args, **kwargs)
        return WorkflowColumn(self, task)

    def op_to_df(self, cols: List[str], op: str, *args: Any, **kwargs: Any) -> WorkflowDataFrame:
        task = self.add(op, *args, **kwargs)

        def get_cols():
            deps: Dict[str, str] = {}
            self._add_dep(deps, task)
            for col in cols:
                value = ExtractColumn(col)
                value = self._add_task(value, deps)
                yield WorkflowColumn(self, value, col)
        return WorkflowDataFrame(*list(get_cols()))

    def show(self, df: WorkflowDataFrame) -> None:
        self.add('show', df)

    def show_col(self, col: WorkflowColumn) -> None:
        self.add('show_col', col)

    def assemble_output(self, *args: Any) -> None:
        task = self.add('assemble_df', *args)
        deps: ParamDict = {}
        self._add_dep(deps, task)
        self._add_task(Output(), deps)

    def add(self, op: str, *args: Any, **kwargs: Any) -> QPDTask:
        deps: ParamDict = {}
        for i in range(len(args)):
            if isinstance(args[i], (QPDTask, QPDTaskWrapper)):
                self._add_dep(deps, args[i])
        args = args[len(deps):]
        task = QPDTask(op, len(deps), 1, *args, **kwargs)
        return self._add_task(task, deps)

    def _extract_df(self, df_name: str) -> WorkflowDataFrame:

        def extract_col(df_name: str, col_name: str) -> WorkflowColumn:
            task = CreateColumn(df_name, col_name)
            task = self._add_task(task)
            return WorkflowColumn(self, task, col_name)
        cols = self.ctx.dfs[df_name].keys()
        return WorkflowDataFrame(*[extract_col(df_name, x) for x in cols])

    def _add_task(self, task, dependencies: Optional[Dict[str, str]]=None) -> QPDTask:
        if dependencies is None:
            pre_add_id = task.pre_add_uuid()
        else:
            pre_add_id = task.pre_add_uuid(dependencies)
        if pre_add_id not in self._tasks:
            name = '_' + str(len(self._spec.tasks))
            task = self._spec.add_task(name, task, dependencies)
            self._tasks[pre_add_id] = task
        return self._tasks[pre_add_id]

    def _add_dep(self, deps: Dict[str, str], obj: Any):
        if isinstance(obj, QPDTask):
            oe = obj.name + '._0'
        elif isinstance(obj, QPDTaskWrapper):
            oe = obj.task.name + '._0'
        else:
            raise ValueError(f'{obj} is invalid')
        deps[f'_{len(deps)}'] = oe