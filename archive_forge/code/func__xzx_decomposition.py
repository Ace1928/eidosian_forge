import numpy as np
import pennylane as qml
from pennylane import math
def _xzx_decomposition(U, wire, return_global_phase=False):
    """Computes the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of Z and X rotations in the form
    :math:`e^{i\\gamma} RX(\\phi) RZ(\\theta) RX(\\lambda)`. (batched operation)

    Args:
        U (tensor): A :math:`2 \\times 2` unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase ``qml.GlobalPhase(-gamma)``
            as the last element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of gates, an ``RX``, an ``RZ`` and
        another ``RX`` gate, which when applied in the order of appearance in the list is
        equivalent to the unitary :math:`U` up to a global phase. If `return_global_phase=True`,
        the global phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])
    >>> decompositions = _xzx_decomposition(U, 0, return_global_phase=True)
    >>> decompositions
    [RX(12.416147693665032, wires=[0]),
     RZ(1.3974974090935608, wires=[0]),
     RX(11.448040119199066, wires=[0]),
     GlobalPhase(1.1759220332464762, wires=[])]

    """
    EPS = 1e-64
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U
    U_det1, gammas = _convert_to_su2(U, return_global_phase=True)
    sum_diagonal_real = math.real(U_det1[:, 0, 0] + U_det1[:, 1, 1])
    sum_off_diagonal_imag = math.imag(U_det1[:, 0, 1] + U_det1[:, 1, 0])
    phi_plus_lambdas_d2 = math.arctan2(-sum_off_diagonal_imag, sum_diagonal_real + EPS)
    diff_diagonal_imag = math.imag(U_det1[:, 0, 0] - U_det1[:, 1, 1])
    diff_off_diagonal_real = math.real(U_det1[:, 0, 1] - U_det1[:, 1, 0])
    phi_minus_lambdas_d2 = math.arctan2(diff_off_diagonal_real, -diff_diagonal_imag + EPS)
    lams = phi_plus_lambdas_d2 - phi_minus_lambdas_d2
    phis = phi_plus_lambdas_d2 + phi_minus_lambdas_d2
    thetas = math.where(math.isclose(math.sin(phi_plus_lambdas_d2), math.zeros_like(phi_plus_lambdas_d2)), 2 * math.arccos(sum_diagonal_real / (2 * math.cos(phi_plus_lambdas_d2) + EPS)), 2 * math.arccos(-sum_off_diagonal_imag / (2 * math.sin(phi_plus_lambdas_d2) + EPS)))
    phis, thetas, lams, gammas = map(math.squeeze, [phis, thetas, lams, gammas])
    phis = phis % (4 * np.pi)
    thetas = thetas % (4 * np.pi)
    lams = lams % (4 * np.pi)
    operations = [qml.RX(lams, wire), qml.RZ(thetas, wire), qml.RX(phis, wire)]
    if return_global_phase:
        operations.append(qml.GlobalPhase(-gammas))
    return operations