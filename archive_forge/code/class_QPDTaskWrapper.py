from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
class QPDTaskWrapper(object):

    def __init__(self, workflow: 'QPDWorkflow', task: QPDTask):
        self._workflow = workflow
        self._task = task

    @property
    def workflow(self) -> 'QPDWorkflow':
        return self._workflow

    @property
    def task(self) -> QPDTask:
        return self._task