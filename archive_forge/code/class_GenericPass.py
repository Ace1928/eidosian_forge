from __future__ import annotations
import logging
import time
from abc import abstractmethod, ABC
from collections.abc import Iterable, Callable, Generator
from typing import Any
from .compilation_status import RunState, PassManagerState, PropertySet
class GenericPass(Task, ABC):
    """Base class of a single pass manager task.

    A pass instance can read and write to the provided :class:`.PropertySet`,
    and may modify the input pass manager IR.
    """

    def __init__(self):
        self.property_set = PropertySet()
        self.requires: Iterable[Task] = []

    def name(self) -> str:
        """Name of the pass."""
        return self.__class__.__name__

    def execute(self, passmanager_ir: PassManagerIR, state: PassManagerState, callback: Callable=None) -> tuple[PassManagerIR, PassManagerState]:
        self.property_set = state.property_set
        if self.requires:
            from .flow_controllers import FlowControllerLinear
            passmanager_ir, state = FlowControllerLinear(self.requires).execute(passmanager_ir=passmanager_ir, state=state, callback=callback)
        run_state = None
        ret = None
        start_time = time.time()
        try:
            if self not in state.workflow_status.completed_passes:
                ret = self.run(passmanager_ir)
                run_state = RunState.SUCCESS
            else:
                run_state = RunState.SKIP
        except Exception:
            run_state = RunState.FAIL
            raise
        finally:
            ret = ret or passmanager_ir
            if run_state != RunState.SKIP:
                running_time = time.time() - start_time
                logger.info('Pass: %s - %.5f (ms)', self.name(), running_time * 1000)
                if callback is not None:
                    callback(task=self, passmanager_ir=ret, property_set=state.property_set, running_time=running_time, count=state.workflow_status.count)
        return (ret, self.update_status(state, run_state))

    def update_status(self, state: PassManagerState, run_state: RunState) -> PassManagerState:
        """Update workflow status.

        Args:
            state: Pass manager state to update.
            run_state: Completion status of current task.

        Returns:
            Updated pass manager state.
        """
        state.workflow_status.previous_run = run_state
        if run_state == RunState.SUCCESS:
            state.workflow_status.count += 1
            state.workflow_status.completed_passes.add(self)
        return state

    @abstractmethod
    def run(self, passmanager_ir: PassManagerIR) -> PassManagerIR:
        """Run optimization task.

        Args:
            passmanager_ir: Qiskit IR to optimize.

        Returns:
            Optimized Qiskit IR.
        """
        pass