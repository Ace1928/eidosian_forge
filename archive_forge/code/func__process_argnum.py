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
def _process_argnum(argnum, tape):
    """Process the argnum keyword argument to ``param_shift_hessian`` from any of ``None``,
    ``int``, ``Sequence[int]``, ``array_like[bool]`` to an ``array_like[bool]``."""
    _trainability_note = 'This may be caused by attempting to differentiate with respect to parameters that are not marked as trainable.'
    if argnum is None:
        argnum = list(range(tape.num_params))
    elif isinstance(argnum, int):
        if argnum >= tape.num_params:
            raise ValueError(f'The index {argnum} exceeds the number of trainable tape parameters ({tape.num_params}). ' + _trainability_note)
        argnum = [argnum]
    if len(qml.math.shape(argnum)) == 1:
        if not qml.math.array(argnum).dtype == bool:
            if qml.math.max(argnum) >= tape.num_params:
                raise ValueError(f'The index {qml.math.max(argnum)} exceeds the number of trainable tape parameters ({tape.num_params}).' + _trainability_note)
            argnum = [i in argnum for i in range(tape.num_params)]
        elif len(argnum) != tape.num_params:
            raise ValueError(f'One-dimensional Boolean array argnum is expected to have as many entries as the tape has trainable parameters ({tape.num_params}), but got {len(argnum)}.' + _trainability_note)
        argnum = qml.math.tensordot(argnum, argnum, axes=0)
    elif not (qml.math.shape(argnum) == (tape.num_params,) * 2 and qml.math.array(argnum).dtype == bool and qml.math.allclose(qml.math.transpose(argnum), argnum)):
        raise ValueError(f'Expected a symmetric 2D Boolean array with shape {(tape.num_params,) * 2} for argnum, but received {argnum}.' + _trainability_note)
    return argnum