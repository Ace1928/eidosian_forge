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
def _pl_op_to_stim(op):
    """Convert PennyLane operation to a Stim operation"""
    try:
        stim_op = _OPERATIONS_MAP[op.name]
        stim_tg = map(str, op.wires)
    except KeyError as e:
        raise qml.DeviceError(f'Operator {op} not supported on default.clifford and does not provide a decomposition.') from e
    if isinstance(op, qml.operation.Channel):
        stim_op += f'({op.parameters[-1]})'
        if op.name == 'PauliError':
            stim_tg = [pauli + wire for pauli, wire in zip(op.parameters[0], stim_tg)]
    return (stim_op, ' '.join(stim_tg))