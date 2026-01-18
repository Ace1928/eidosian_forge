from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def _all_zero_grad(tape):
    """Auxiliary function to return zeros for the all-zero gradient case."""
    list_zeros = []
    par_shapes = [qml.math.shape(p) for p in tape.get_parameters()]
    for m in tape.measurements:
        shape = (2 ** len(m.wires),) if isinstance(m, ProbabilityMP) else ()
        if len(tape.trainable_params) == 1:
            sub_list_zeros = qml.math.zeros(par_shapes[0] + shape)
        else:
            sub_list_zeros = tuple((qml.math.zeros(sh + shape) for sh in par_shapes))
        list_zeros.append(sub_list_zeros)
    if tape.shots.has_partitioned_shots:
        if len(tape.measurements) == 1:
            return ([], lambda _: tuple((list_zeros[0] for _ in range(tape.shots.num_copies))))
        return ([], lambda _: tuple((tuple(list_zeros) for _ in range(tape.shots.num_copies))))
    if len(tape.measurements) == 1:
        return ([], lambda _: list_zeros[0])
    return ([], lambda _: tuple(list_zeros))