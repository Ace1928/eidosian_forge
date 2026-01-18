from typing import Sequence, Callable
import itertools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import (
from .finite_difference import finite_diff
from .general_shift_rules import generate_shifted_tapes, process_shifts
from .gradient_transform import _no_trainable_grad
from .parameter_shift import _get_operation_recipe, expval_param_shift
def _expand_transform_param_shift_cv(tape: qml.tape.QuantumTape, dev, argnum=None, shifts=None, gradient_recipes=None, fallback_fn=finite_diff, f0=None, force_order2=False) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Expand function to be applied before parameter shift CV."""
    expanded_tape = expand_invalid_trainable(tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]
    return ([expanded_tape], null_postprocessing)