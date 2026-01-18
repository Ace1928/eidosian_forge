import numpy as np
import pennylane as qml
from pennylane import math
def _zxz_decomposition(U, wire, return_global_phase=False):
    """Compute the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of X and Z rotations in the form
    :math:`e^{i\\alpha} RZ(\\phi) RY(\\theta) RZ(\\psi)`. (batched operation)

    Args:
        U (array[complex]): A :math:`2 \\times 2` unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase as a
            ``qml.GlobalPhase(-alpha)`` as the last element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of gates, an ``RZ``, an ``RX`` and
            another ``RZ`` gate, which when applied in the order of appearance in the list is
            equivalent to the unitary :math:`U` up to a global phase. If `return_global_phase=True`,
            the global phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])
    >>> decompositions = _zxz_decomposition(U, 0, return_global_phase=True)
    >>> decompositions
    [RZ(10.753478981934784, wires=[0]),
     RX(1.1493817777940707, wires=[0]),
     RZ(3.3038544749132295, wires=[0]),
     GlobalPhase(1.1759220332464762, wires=[])]

    """
    EPS = 1e-64
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U
    U_det1, alphas = _convert_to_su2(U, return_global_phase=True)
    phis_plus_psis = math.arctan2(-math.imag(U_det1[:, 0, 0]), math.real(U_det1[:, 0, 0]) + EPS)
    phis_minus_psis = math.arctan2(-math.real(U_det1[:, 0, 1]), -math.imag(U_det1[:, 0, 1]) + EPS)
    phis = phis_plus_psis + phis_minus_psis
    psis = phis_plus_psis - phis_minus_psis
    thetas = math.where(math.isclose(math.sin(phis_plus_psis), math.zeros_like(phis_plus_psis)), math.real(U_det1[:, 0, 0]) / (math.cos(phis_plus_psis) + EPS), -math.imag(U_det1[:, 0, 0]) / (math.sin(phis_plus_psis) + EPS))
    thetas = qml.math.clip(thetas, -1.0, 1.0)
    thetas = 2 * math.arccos(thetas)
    phis, thetas, psis, alphas = map(math.squeeze, [phis, thetas, psis, alphas])
    phis = phis % (4 * np.pi)
    thetas = thetas % (4 * np.pi)
    psis = psis % (4 * np.pi)
    operations = [qml.RZ(psis, wire), qml.RX(thetas, wire), qml.RZ(phis, wire)]
    if return_global_phase:
        operations.append(qml.GlobalPhase(-alphas))
    return operations