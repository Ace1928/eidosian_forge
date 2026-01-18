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
def _sample_classical_shadow(self, meas, stim_circuit, shots, seed):
    """Measures classical shadows from the state of simulator device"""
    meas_seed = meas.seed or seed
    meas_wire = stim_circuit.num_qubits
    bits = []
    recipes = np.random.RandomState(meas_seed).randint(3, size=(shots, meas_wire))
    for recipe in recipes:
        bits.append([self._measure_single_sample(stim_circuit, [rec + 1], idx, meas_wire) for idx, rec in enumerate(recipe)])
    return (np.asarray(bits, dtype=int), np.asarray(recipes, dtype=int))