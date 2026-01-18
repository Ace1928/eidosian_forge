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
def _sample_variance(self, meas, stim_circuit, shots, seed):
    """Measure the variance with respect to samples from simulator device."""
    meas_op = meas.obs
    meas_obs = qml.operation.convert_to_opmath(meas_op)
    meas_obs1 = meas_obs.simplify()
    meas_obs2 = (meas_obs1 ** 2).simplify()
    return self._sample_expectation(qml.expval(meas_obs2), stim_circuit, shots, seed) - self._sample_expectation(qml.expval(meas_obs1), stim_circuit, shots, seed) ** 2