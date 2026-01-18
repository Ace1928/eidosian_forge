from functools import wraps
from pennylane.devices import DefaultExecutionConfig, Device
from pennylane.tape import QuantumScript
from ..qubit.sampling import get_num_shots_and_executions
def _track_compute_derivatives(untracked_compute_derivatives):
    """Adds default tracking to a ``compute_derivatives`` method."""

    @wraps(untracked_compute_derivatives)
    def compute_derivatives(self, circuits, execution_config=DefaultExecutionConfig):
        if self.tracker.active:
            if isinstance(circuits, QuantumScript):
                derivatives = 1
            else:
                derivatives = len(circuits)
            self.tracker.update(derivative_batches=1, derivatives=derivatives)
            self.tracker.record()
        return untracked_compute_derivatives(self, circuits, execution_config)
    return compute_derivatives