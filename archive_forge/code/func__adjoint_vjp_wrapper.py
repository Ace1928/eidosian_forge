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
def _adjoint_vjp_wrapper(c, t, rng=None, prng_key=None, debugger=None):
    state, is_state_batched = get_final_state(c, debugger=debugger)
    vjp = adjoint_vjp(c, t, state=state)
    res = measure_final_state(c, state, is_state_batched, rng=rng, prng_key=prng_key)
    return (res, vjp)