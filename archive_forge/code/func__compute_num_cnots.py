import numpy as np
import pennylane as qml
from pennylane import math
from .single_qubit_unitary import one_qubit_decomposition
def _compute_num_cnots(U):
    """Compute the number of CNOTs required to implement a U in SU(4). This is based on
    the trace of

    .. math::

        \\gamma(U) = (E^\\dag U E) (E^\\dag U E)^T,

    and follows the arguments of this paper: https://arxiv.org/abs/quant-ph/0308045.
    """
    u = math.dot(Edag, math.dot(U, E))
    gammaU = math.dot(u, math.T(u))
    trace = math.trace(gammaU)
    if math.allclose(trace, 4, atol=1e-07) or math.allclose(trace, -4, atol=1e-07):
        return 0
    evs = math.linalg.eigvals(gammaU)
    sorted_evs = math.sort(math.imag(evs))
    if math.allclose(trace, 0j, atol=1e-07) and math.allclose(sorted_evs, [-1, -1, 1, 1]):
        return 1
    if math.allclose(math.imag(trace), 0.0, atol=1e-07):
        return 2
    return 3