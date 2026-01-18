from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane import transform
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from pennylane.transforms.tape_expand import expand_invalid_trainable
from .finite_difference import _processing_fn, finite_diff_coeffs
from .gradient_transform import (
from .general_shift_rules import generate_multishifted_tapes
def _single_shot_batch_result(results):
    """Auxiliary function for post-processing one batch of results corresponding to finite
        shots or a single component of a shot vector"""
    r0, results = (results[0], results[1:]) if extract_r0 else (f0, results)
    num_measurements = len(tape.measurements)
    if num_measurements == 1:
        grads = 0
        for rep, _coeffs in enumerate(all_coeffs):
            res = list(results[rep * tapes_per_grad:(rep + 1) * tapes_per_grad])
            if r0 is not None:
                res.insert(0, r0)
            res = qml.math.stack(res)
            grads = qml.math.tensordot(qml.math.convert_like(_coeffs, res), res, axes=[[0], [0]]) + grads
        grads = grads * (1 / num_directions)
        if num_trainable_params == 1:
            return qml.math.convert_like(grads[0], res)
        return tuple((qml.math.convert_like(g, res) for g in grads))
    grads = []
    for i in range(num_measurements):
        grad = 0
        for rep, _coeffs in enumerate(all_coeffs):
            res = [r[i] for r in results[rep * tapes_per_grad:(rep + 1) * tapes_per_grad]]
            if r0 is not None:
                res.insert(0, r0[i])
            res = qml.math.stack(res)
            grad = qml.math.tensordot(qml.math.convert_like(_coeffs, res), res, axes=[[0], [0]]) + grad
        grad = grad / num_directions
        grads.append(tuple((qml.math.convert_like(g, grad) for g in grad)))
    if num_trainable_params == 1:
        return tuple((g[0] for g in grads))
    return tuple(grads)