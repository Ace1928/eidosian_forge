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
def _create_variance_proc_fn(tape, var_mask, var_indices, pdA_fn, pdA2_fn, tape_boundary, non_involutory_indices):
    """Auxiliary function to define the processing function for computing the
    derivative of variances using the parameter-shift rule.

    Args:
        var_mask (list): The mask of variance measurements in the measurement queue.
        var_indices (list): The indices of variance measurements in the measurement queue.
        pdA_fn (callable): The function required to evaluate the analytic derivative of <A>.
        pdA2_fn (callable): If not None, non-involutory observables are
            present; the partial derivative of <A^2> may be non-zero. Here, we
            calculate the analytic derivatives of the <A^2> observables.
        tape_boundary (callable): the number of first derivative tapes used to
            determine the number of results to post-process later
        non_involutory_indices (list): the indices in the measurement queue of all non-involutory
            observables
    """

    def processing_fn(results):
        f0 = results[0]
        pdA = pdA_fn(results[int(not pdA_fn.first_result_unshifted):tape_boundary])
        pdA2 = _get_pdA2(results[tape_boundary:], tape, pdA2_fn, non_involutory_indices, var_indices)
        if tape.shots.has_partitioned_shots:
            final_res = []
            for idx_shot_comp in range(tape.shots.num_copies):
                f0_comp = f0[idx_shot_comp]
                pdA_comp = pdA[idx_shot_comp]
                pdA2_comp = pdA2 if isinstance(pdA2, int) else pdA2[idx_shot_comp]
                r = _single_variance_gradient(tape, var_mask, pdA2_comp, f0_comp, pdA_comp)
                final_res.append(r)
            return tuple(final_res)
        return _single_variance_gradient(tape, var_mask, pdA2, f0, pdA)
    return processing_fn