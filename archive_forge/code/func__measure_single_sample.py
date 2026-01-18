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
@staticmethod
def _measure_single_sample(stim_ct, meas_ops, meas_idx, meas_wire):
    """Sample a single qubit Pauli measurement from a stim circuit"""
    stim_sm = stim.TableauSimulator()
    stim_sm.do_circuit(stim_ct)
    return stim_sm.measure_observable(stim.PauliString([0] * meas_idx + meas_ops + [0] * (meas_wire - meas_idx - 1)))