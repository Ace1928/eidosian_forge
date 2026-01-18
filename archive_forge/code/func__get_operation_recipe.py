from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .finite_difference import finite_diff
from .general_shift_rules import (
from .gradient_transform import (
def _get_operation_recipe(tape, t_idx, shifts, order=1):
    """Utility function to return the parameter-shift rule
    of the operation corresponding to trainable parameter
    t_idx on tape.

    Args:
        tape (.QuantumTape): Tape containing the operation to differentiate
        t_idx (int): Parameter index of the operation to differentiate within the tape
        shifts (Sequence[float or int]): Shift values to use if no static ``grad_recipe`` is
            provided by the operation to differentiate
        order (int): Order of the differentiation

    This function performs multiple attempts to obtain the recipe:

    - If the operation has a custom :attr:`~.grad_recipe` defined, it is used.

    - If :attr:`.parameter_frequencies` yields a result, the frequencies are
      used to construct the general parameter-shift rule via
      :func:`.generate_shift_rule`.
      Note that by default, the generator is used to compute the parameter frequencies
      if they are not provided by a custom implementation.

    That is, the order of precedence is :meth:`~.grad_recipe`, custom
    :attr:`~.parameter_frequencies`, and finally :meth:`.generator` via the default
    implementation of the frequencies.

    If order is set to 2, the rule for the second-order derivative is obtained instead.
    """
    if order not in {1, 2}:
        raise NotImplementedError('_get_operation_recipe only is implemented for orders 1 and 2.')
    op, _, p_idx = tape.get_operation(t_idx)
    op_recipe = _process_op_recipe(op, p_idx, order)
    if op_recipe is not None:
        return op_recipe
    try:
        frequencies = op.parameter_frequencies[p_idx]
    except qml.operation.ParameterFrequenciesUndefinedError as e:
        raise qml.operation.OperatorPropertyUndefined(f'The operation {op.name} does not have a grad_recipe, parameter_frequencies or a generator defined. No parameter shift rule can be applied.') from e
    coeffs, shifts = qml.gradients.generate_shift_rule(frequencies, shifts=shifts, order=order).T
    mults = np.ones_like(coeffs)
    return qml.math.stack([coeffs, mults, shifts]).T