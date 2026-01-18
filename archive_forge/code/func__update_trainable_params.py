import contextlib
import pennylane as qml
from pennylane.operation import (
def _update_trainable_params(tape):
    params = tape.get_parameters(trainable_only=False)
    tape.trainable_params = qml.math.get_trainable_indices(params)