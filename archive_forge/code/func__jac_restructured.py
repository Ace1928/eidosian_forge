from functools import reduce
import numpy as np
import tensorflow as tf
import pennylane as qml
from pennylane.measurements import SampleMP, StateMP
from .tensorflow import (
def _jac_restructured(jacs, tapes):
    """
    Reconstruct the nested tuple structure of the jacobian of a list of tapes
    """
    start = 0
    jacs_nested = []
    for tape in tapes:
        num_meas = len(tape.measurements)
        num_params = len(tape.trainable_params)
        tape_jacs = tuple(jacs[start:start + num_meas * num_params])
        tape_jacs = tuple((tuple(tape_jacs[i * num_params:(i + 1) * num_params]) for i in range(num_meas)))
        while isinstance(tape_jacs, tuple) and len(tape_jacs) == 1:
            tape_jacs = tape_jacs[0]
        jacs_nested.append(tape_jacs)
        start += num_meas * num_params
    return tuple(jacs_nested)