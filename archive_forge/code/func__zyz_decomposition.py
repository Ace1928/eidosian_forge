import numpy as np
import pennylane as qml
from pennylane import math
def _zyz_decomposition(U, wire, return_global_phase=False):
    """Compute the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of Z and Y rotations in the form
    :math:`e^{i\\alpha} RZ(\\omega) RY(\\theta) RZ(\\phi)`. (batched operation)

    .. warning::

        When used with ``jax.jit``, all unitaries will be converted to :class:`.Rot` gates,
        including those that are diagonal.

    Args:
        U (tensor): A :math:`2 \\times 2` unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase ``qml.GlobalPhase(-alpha)``
            as the last element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of gates, an ``RZ``, an ``RY`` and another ``RZ`` gate,
            which when applied in the order of appearance in the list is equivalent to the
            unitary :math:`U` up to a global phase. If `return_global_phase=True`, the global
            phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])
    >>> decompositions = _zyz_decomposition(U, 0, return_global_phase=True)
    >>> decompositions
    [RZ(12.32427531154459, wires=[0]),
     RY(1.1493817771511354, wires=[0]),
     RZ(1.733058145303424, wires=[0]),
     GlobalPhase(1.1759220332464762, wires=[])]

    """
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U
    U_det1, alphas = _convert_to_su2(U, return_global_phase=True)
    phis, thetas, omegas = _zyz_get_rotation_angles(U_det1)
    operations = [qml.RZ(phis, wire), qml.RY(thetas, wire), qml.RZ(omegas, wire)]
    if return_global_phase:
        alphas = math.squeeze(alphas)
        operations.append(qml.GlobalPhase(-alphas))
    return operations