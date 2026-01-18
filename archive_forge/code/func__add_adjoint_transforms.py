from dataclasses import replace
from functools import partial
from numbers import Number
from typing import Union, Callable, Tuple, Optional, Sequence
import concurrent.futures
import inspect
import logging
import numpy as np
import pennylane as qml
from pennylane.ops.op_math.condition import Conditional
from pennylane.measurements.mid_measure import MidMeasureMP
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from . import Device
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .qubit.simulate import simulate, get_final_state, measure_final_state
from .qubit.adjoint_jacobian import adjoint_jacobian, adjoint_vjp, adjoint_jvp
def _add_adjoint_transforms(program: TransformProgram, device_vjp=False) -> None:
    """Private helper function for ``preprocess`` that adds the transforms specific
    for adjoint differentiation.

    Args:
        program (TransformProgram): where we will add the adjoint differentiation transforms

    Side Effects:
        Adds transforms to the input program.

    """
    name = 'adjoint + default.qubit'
    program.add_transform(no_sampling, name=name)
    program.add_transform(decompose, stopping_condition=adjoint_ops, name=name)
    program.add_transform(validate_observables, adjoint_observables, name=name)
    program.add_transform(validate_measurements, name=name)
    program.add_transform(adjoint_state_measurements, device_vjp=device_vjp)
    program.add_transform(qml.transforms.broadcast_expand)
    program.add_transform(validate_adjoint_trainable_params)