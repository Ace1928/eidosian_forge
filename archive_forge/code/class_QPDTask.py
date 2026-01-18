from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
class QPDTask(TaskSpec):

    def __init__(self, op_name: str, input_n: int=0, output_n: int=1, *args: Any, **kwargs: Any):
        self._op_name = op_name
        self._args = args
        self._kwargs = kwargs
        assert_or_throw(output_n <= 1, NotImplementedError('multiple output tasks'))
        inputs = [InputSpec('_' + str(i), object, nullable=False) for i in range(input_n)]
        outputs = [OutputSpec('_' + str(i), object, nullable=False) for i in range(output_n)]
        super().__init__(configs=None, inputs=inputs, outputs=outputs, func=self.execute, deterministic=True, lazy=False)

    def pre_add_uuid(self, *args: Any, **kwargs) -> str:
        return to_uuid(self.configs, self.inputs, self.outputs, self._op_name, self._args, self._kwargs, args, kwargs)

    def __uuid__(self) -> str:
        return to_uuid(self.configs, self.inputs, self.outputs, self.node_spec, self._op_name, self._args, self._kwargs)

    def execute(self, ctx: TaskContext) -> None:
        ctx.ensure_all_ready()
        op = ctx.workflow_context.qpd_engine
        res = op(self._op_name, *list(ctx.inputs.values()), *self._args, **self._kwargs)
        ctx.outputs['_0'] = res