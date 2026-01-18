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
def _sample_expectation(self, meas, stim_circuit, shots, seed):
    """Measure the expectation value with respect to samples from simulator device."""
    meas_op = meas.obs
    samples, coeffs = self._measure_observable_sample(meas_op, stim_circuit, shots, seed)
    if isinstance(meas_op, BasisStateProjector):
        matches = np.where((samples[0] == meas_op.data[0]).all(axis=1))[0]
        return len(matches) / shots
    expecs = [qml.math.mean(qml.math.power([-1] * shots, qml.math.sum(sample, axis=1))) for sample in samples]
    return qml.math.dot(coeffs, expecs)