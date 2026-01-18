import sys
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Callable, List, Optional, no_type_check
from adagio.instances import TaskContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec
from triad import ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from fugue._utils.exception import (
from fugue.collections.partition import PartitionSpec
from fugue.collections.yielded import PhysicalYielded
from fugue.dataframe import DataFrame, DataFrames
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.exceptions import FugueWorkflowError
from fugue.execution import ExecutionEngine
from fugue.extensions.creator.convert import _to_creator
from fugue.extensions.outputter.convert import _to_outputter
from fugue.extensions.processor.convert import _to_processor
from fugue.rpc.base import RPCServer
from fugue.workflow._checkpoint import Checkpoint, StrongCheckpoint
from fugue.workflow._workflow_context import FugueWorkflowContext
def _get_dependency_uuid(self) -> Any:
    if self._dependency_uuid is not None:
        return self._dependency_uuid
    values: List[Any] = []
    for k, v in self.node_spec.dependency.items():
        t = v.split('.', 1)
        assert_or_throw(len(t) == 2)
        values.append(k)
        values.append(t[1])
        task = self.parent_workflow.tasks[t[0]]
        values.append(task.__uuid__())
    self._dependency_uuid = to_uuid(values)
    return self._dependency_uuid