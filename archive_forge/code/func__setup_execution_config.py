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
def _setup_execution_config(self, execution_config: ExecutionConfig) -> ExecutionConfig:
    """This is a private helper for ``preprocess`` that sets up the execution config.

        Args:
            execution_config (ExecutionConfig)

        Returns:
            ExecutionConfig: a preprocessed execution config

        """
    updated_values = {}
    if execution_config.gradient_method == 'best':
        updated_values['gradient_method'] = None
    updated_values['use_device_jacobian_product'] = False
    if execution_config.grad_on_execution is None:
        updated_values['grad_on_execution'] = False
    updated_values['device_options'] = dict(execution_config.device_options)
    if 'max_workers' not in updated_values['device_options']:
        updated_values['device_options']['max_workers'] = self._max_workers
    if 'rng' not in updated_values['device_options']:
        updated_values['device_options']['rng'] = self._rng
    if 'tableau' not in updated_values['device_options']:
        updated_values['device_options']['tableau'] = self._tableau
    return replace(execution_config, **updated_values)