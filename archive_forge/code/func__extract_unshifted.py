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
def _extract_unshifted(recipe, at_least_one_unshifted, f0, gradient_tapes, tape):
    """Exctract the unshifted term from a gradient recipe, if it is present.

    Returns:
        array_like[float]: The reduced recipe without the unshifted term.
        bool: The updated flag whether an unshifted term was found for any of the recipes.
        float or None: The coefficient of the unshifted term. None if no such term was present.

    This assumes that there will be at most one unshifted term in the recipe (others simply are
    not extracted) and that it comes first if there is one.
    """
    first_c, first_m, first_s = recipe[0]
    if first_s == 0 and first_m == 1:
        if not at_least_one_unshifted and f0 is None:
            gradient_tapes.insert(0, tape)
        unshifted_coeff = first_c
        at_least_one_unshifted = True
        recipe = recipe[1:]
    else:
        unshifted_coeff = None
    return (recipe, at_least_one_unshifted, unshifted_coeff)