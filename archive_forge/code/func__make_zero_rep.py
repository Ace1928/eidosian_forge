from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .finite_difference import finite_diff
from .general_shift_rules import (
from .gradient_transform import (
def _make_zero_rep(g, single_measure, has_partitioned_shots, par_shapes=None):
    """Create a zero-valued gradient entry adapted to the measurements and shot_vector
    of a gradient computation, where g is a previously computed non-zero gradient entry.

    Args:
        g (tensor_like): Gradient entry that was computed for a different parameter, from which
            we inherit the shape and data type of the zero-valued entry to create
        single_measure (bool): Whether the differentiated function returned a single measurement.
        has_partitioned_shots (bool): Whether the differentiated function used a shot vector.
        par_shapes (tuple(tuple)): Shapes of the parameter for which ``g`` is the gradient entry,
            and of the parameter for which to create a zero-valued gradient entry, in this order.

    Returns:
        tensor_like or tuple(tensor_like) or tuple(tuple(tensor_like)): Zero-valued gradient entry
        similar to the non-zero gradient entry ``g``, potentially adapted to differences between
        parameter shapes if ``par_shapes`` were provided.

    """
    cut_dims, par_shape = (len(par_shapes[0]), par_shapes[1]) if par_shapes else (0, ())
    if par_shapes is None:
        zero_entry = qml.math.zeros_like
    else:

        def zero_entry(grad_entry):
            """Create a gradient entry that is zero and has the correctly modified shape."""
            new_shape = par_shape + qml.math.shape(grad_entry)[cut_dims:]
            return qml.math.zeros(new_shape, like=grad_entry)
    if single_measure and (not has_partitioned_shots):
        return zero_entry(g)
    if single_measure or not has_partitioned_shots:
        return tuple(map(zero_entry, g))
    return tuple((tuple(map(zero_entry, shot_comp_g)) for shot_comp_g in g))