from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import ray
from .ref_bundle import RefBundle
from ray._raylet import ObjectRefGenerator
from ray.data._internal.execution.interfaces.execution_options import (
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.logical.interfaces import Operator
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
class OpTask(ABC):
    """Abstract class that represents a task that is created by an PhysicalOperator.

    The task can be either a regular task or an actor task.
    """

    def __init__(self, task_index: int):
        self._task_index = task_index

    def task_index(self) -> int:
        """Return the index of the task."""
        return self._task_index

    @abstractmethod
    def get_waitable(self) -> Waitable:
        """Return the ObjectRef or ObjectRefGenerator to wait on."""
        pass