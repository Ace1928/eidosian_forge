from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
class ExtractColumn(QPDTask):

    def __init__(self, name: str):
        super().__init__('extract_col', 1, 1, name)
        self._name = name

    def execute(self, ctx: TaskContext) -> None:
        op = ctx.workflow_context.qpd_engine
        res = op(self._op_name, ctx.inputs['_0'], self._name)
        ctx.outputs['_0'] = res