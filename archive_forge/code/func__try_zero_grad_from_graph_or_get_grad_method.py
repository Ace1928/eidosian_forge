from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def _try_zero_grad_from_graph_or_get_grad_method(tape, param_index, use_graph=True):
    """Gets the gradient method of a parameter. If use_graph=True, analyze the
    circuit graph to find if the parameter has zero gradient.

    Args:
        tape (`~.QuantumScript`): the tape to analyze
        param_index (int): the index of the parameter to analyze
        use_graph (bool): whether to use the circuit graph to find if
            a parameter has zero gradient

    """
    par_info = tape._par_info[param_index]
    if use_graph:
        op_or_mp = tape[par_info['op_idx']]
        if not any((tape.graph.has_path(op_or_mp, mp) for mp in tape.measurements)):
            return '0'
    return par_info['op'].grad_method