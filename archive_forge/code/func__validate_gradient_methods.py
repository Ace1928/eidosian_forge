from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def _validate_gradient_methods(tape, method, diff_methods):
    """Validates if the gradient method requested is supported by the trainable
    parameters of a tape, and returns the allowed parameter gradient methods."""
    nondiff_params = [tape.trainable_params[idx] for idx, m in diff_methods.items() if m is None]
    if nondiff_params:
        raise ValueError(f'Cannot differentiate with respect to parameter(s) {nondiff_params}')
    numeric_params = [tape.trainable_params[idx] for idx, m in diff_methods.items() if m == 'F']
    if method == 'analytic' and numeric_params:
        raise ValueError(f'The analytic gradient method cannot be used with the parameter(s) {numeric_params}.')