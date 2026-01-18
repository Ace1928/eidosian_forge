import numpy as np
import pennylane as qml
from pennylane.devices import DefaultQubitLegacy
from pennylane.pulse import ParametrizedEvolution
from pennylane.typing import TensorLike
def _apply_parametrized_evolution(self, state: TensorLike, operation: ParametrizedEvolution):
    if 2 * len(operation.wires) > self.num_wires and (not operation.hyperparameters['complementary']):
        return self._evolve_state_vector_under_parametrized_evolution(state, operation)
    return self._apply_operation(state, operation)