from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
def _compute_cfim(p, dp):
    """Computes the (num_params, num_params) classical fisher information matrix from the probabilities and its derivatives
    I.e. it computes :math:`classical_fisher_{ij} = \\sum_\\ell (\\partial_i p_\\ell) (\\partial_i p_\\ell) / p_\\ell`
    """
    nonzeros_p = qml.math.where(p > 0, p, qml.math.ones_like(p))
    one_over_p = qml.math.where(p > 0, qml.math.ones_like(p), qml.math.zeros_like(p))
    one_over_p = one_over_p / nonzeros_p
    dp = qml.math.cast_like(dp, p)
    dp = qml.math.reshape(dp, (len(p), -1))
    dp_over_p = qml.math.transpose(dp) * one_over_p
    return dp_over_p @ dp