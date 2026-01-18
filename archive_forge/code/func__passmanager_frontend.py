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
@abstractmethod
def _passmanager_frontend(self, input_program: Any, **kwargs) -> PassManagerIR:
    """Convert input program into pass manager IR.

        Args:
            in_program: Input program.

        Returns:
            Pass manager IR.
        """
    pass