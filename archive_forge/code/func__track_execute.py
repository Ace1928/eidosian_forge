from functools import wraps
from pennylane.devices import DefaultExecutionConfig, Device
from pennylane.tape import QuantumScript
from ..qubit.sampling import get_num_shots_and_executions
def _track_execute(untracked_execute):
    """Adds default tracking to an execute method."""

    @wraps(untracked_execute)
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        results = untracked_execute(self, circuits, execution_config)
        if isinstance(circuits, QuantumScript):
            batch = (circuits,)
            batch_results = (results,)
        else:
            batch = circuits
            batch_results = results
        if self.tracker.active:
            self.tracker.update(batches=1)
            self.tracker.record()
            for r, c in zip(batch_results, batch):
                qpu_executions, shots = get_num_shots_and_executions(c)
                if c.shots:
                    self.tracker.update(simulations=1, executions=qpu_executions, results=r, shots=shots, resources=c.specs['resources'])
                else:
                    self.tracker.update(simulations=1, executions=qpu_executions, results=r, resources=c.specs['resources'])
                self.tracker.record()
        return results
    return execute