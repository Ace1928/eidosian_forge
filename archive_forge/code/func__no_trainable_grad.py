from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def _no_trainable_grad(tape):
    """Auxiliary function that returns correctly formatted gradients when there
    are no trainable parameters."""
    warnings.warn(_no_trainable_grad_warning)
    if tape.shots.has_partitioned_shots:
        if len(tape.measurements) == 1:
            return ([], lambda _: tuple((qml.math.zeros([0]) for _ in range(tape.shots.num_copies))))
        return ([], lambda _: tuple((tuple((qml.math.zeros([0]) for _ in range(len(tape.measurements)))) for _ in range(tape.shots.num_copies))))
    if len(tape.measurements) == 1:
        return ([], lambda _: qml.math.zeros([0]))
    return ([], lambda _: tuple((qml.math.zeros([0]) for _ in range(len(tape.measurements)))))