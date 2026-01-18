from dataclasses import replace
from functools import partial
from typing import Union, Tuple, Sequence
import concurrent.futures
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import (
from pennylane.ops.qubit.observables import BasisStateProjector
from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .default_qubit import accepted_sample_measurement
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
def _pl_obs_to_linear_comb(meas_op):
    """Convert a PennyLane observable to a linear combination of Pauli strings"""
    meas_obs = qml.operation.convert_to_opmath(meas_op)
    meas_rep = meas_obs.pauli_rep
    if isinstance(meas_obs, (qml.Hermitian, BasisStateProjector)):
        meas_rep = qml.pauli_decompose(meas_obs.matrix(), wire_order=meas_obs.wires, pauli=True)
    if meas_rep is None:
        raise NotImplementedError(f"default.clifford doesn't support expectation value calculation with {type(meas_op).__name__} at the moment.")
    coeffs, paulis = (np.array(list(meas_rep.values())), [])
    meas_op_wires = list(meas_op.wires)
    for pw in meas_rep:
        p_wire, p_word = (pw.keys(), pw.values())
        if not p_word:
            r_wire, r_word = (meas_op_wires[:1], 'I')
        else:
            r_wire = sorted(p_wire, key=meas_op_wires.index)
            r_word = ''.join(map(pw.get, r_wire))
        paulis.append((r_word, r_wire))
    return (coeffs, paulis)