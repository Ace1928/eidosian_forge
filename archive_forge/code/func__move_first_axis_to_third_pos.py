from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def _move_first_axis_to_third_pos(grads, first_axis_size, second_axis_size, third_axis_size):
    """Transpose the first two axes of an iterable of iterables, returning
    a tuple of tuples."""
    if first_axis_size == 1:
        return tuple((tuple((grads[0][i][j] for j in range(third_axis_size))) for i in range(second_axis_size)))
    return tuple((tuple((tuple((grads[k][i][j] for k in range(first_axis_size))) for j in range(third_axis_size))) for i in range(second_axis_size)))