import itertools as it
import warnings
from functools import partial
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP, StateMP, VarianceMP
from pennylane.transforms import transform
from .general_shift_rules import (
from .gradient_transform import find_and_validate_gradient_methods
from .parameter_shift import _get_operation_recipe
from .hessian_transform import _process_jacs
def _no_trainable_hessian(tape):
    warnings.warn(_no_trainable_hessian_warning)
    if len(tape.measurements) == 1:
        return ([], lambda _: qml.math.zeros((0,)))
    return ([], lambda _: tuple((qml.math.zeros((0,)) for _ in tape.measurements)))