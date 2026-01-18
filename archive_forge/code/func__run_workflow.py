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
def _run_workflow(program: Any, pass_manager: BasePassManager, **kwargs) -> Any:
    """Run single program optimization with a pass manager.

    Args:
        program: Arbitrary program to optimize.
        pass_manager: Pass manager with scheduled passes.
        **kwargs: Keyword arguments for IR conversion.

    Returns:
        Optimized program.
    """
    flow_controller = pass_manager.to_flow_controller()
    initial_status = WorkflowStatus()
    passmanager_ir = pass_manager._passmanager_frontend(input_program=program, **kwargs)
    passmanager_ir, final_state = flow_controller.execute(passmanager_ir=passmanager_ir, state=PassManagerState(workflow_status=initial_status, property_set=PropertySet()), callback=kwargs.get('callback', None))
    pass_manager.property_set = final_state.property_set
    out_program = pass_manager._passmanager_backend(passmanager_ir=passmanager_ir, in_program=program, **kwargs)
    return out_program