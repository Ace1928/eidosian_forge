from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from itertools import chain
from typing import Any
import dill
from qiskit.utils.parallel import parallel_map
from .base_tasks import Task, PassManagerIR
from .exceptions import PassManagerError
from .flow_controllers import FlowControllerLinear
from .compilation_status import PropertySet, WorkflowStatus, PassManagerState
def _flatten_tasks(self, elements: Iterable | Task) -> Iterable:
    """A helper method to recursively flatten a nested task chain."""
    if not isinstance(elements, Iterable):
        return [elements]
    return chain(*map(self._flatten_tasks, elements))