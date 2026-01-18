from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def _find_gradient_methods(tape, trainable_param_indices, use_graph=True):
    """Returns a dictionary with gradient information of each trainable parameter."""
    return {idx: _try_zero_grad_from_graph_or_get_grad_method(tape, tape.trainable_params[idx], use_graph) for idx in trainable_param_indices}