from functools import wraps
from pennylane.devices import DefaultExecutionConfig, Device
from pennylane.tape import QuantumScript
from ..qubit.sampling import get_num_shots_and_executions
def _track_execute_and_compute_vjp(untracked_execute_and_compute_vjp):
    """Adds default trakcing to a ``execute_and_compute_vjp`` method."""

    @wraps(untracked_execute_and_compute_vjp)
    def execute_and_compute_vjp(self, circuits, cotangents, execution_config=DefaultExecutionConfig):
        if self.tracker.active:
            batch = (circuits,) if isinstance(circuits, QuantumScript) else circuits
            for c in batch:
                self.tracker.update(resources=c.specs['resources'])
            self.tracker.update(execute_and_vjp_batches=1, executions=len(batch), vjps=len(batch))
            self.tracker.record()
        return untracked_execute_and_compute_vjp(self, circuits, cotangents, execution_config)
    return execute_and_compute_vjp