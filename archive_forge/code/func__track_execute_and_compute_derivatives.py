from functools import wraps
from pennylane.devices import DefaultExecutionConfig, Device
from pennylane.tape import QuantumScript
from ..qubit.sampling import get_num_shots_and_executions
def _track_execute_and_compute_derivatives(untracked_execute_and_compute_derivatives):
    """Adds default tracking to a ``execute_and_compute_derivatives`` method."""

    @wraps(untracked_execute_and_compute_derivatives)
    def execute_and_compute_derivatives(self, circuits, execution_config=DefaultExecutionConfig):
        if self.tracker.active:
            batch = (circuits,) if isinstance(circuits, QuantumScript) else circuits
            for c in batch:
                self.tracker.update(resources=c.specs['resources'])
            self.tracker.update(execute_and_derivative_batches=1, executions=len(batch), derivatives=len(batch))
            self.tracker.record()
        return untracked_execute_and_compute_derivatives(self, circuits, execution_config)
    return execute_and_compute_derivatives