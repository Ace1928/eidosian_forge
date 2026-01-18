import numpy as np
import pennylane as qml
from pennylane import math
from .single_qubit_unitary import one_qubit_decomposition
def _decomposition_3_cnots(U, wires):
    """The most general form of this decomposition is U = (A \\otimes B) V (C \\otimes D),
    where V is as depicted in the circuit below:
     -╭U- = -C--╭X--RZ(d)--╭C---------╭X--A-
     -╰U- = -D--╰C--RY(b)--╰X--RY(a)--╰C--B-
    """
    swap_U = np.exp(1j * np.pi / 4) * math.dot(math.cast_like(SWAP, U), U)
    u = math.dot(Edag, math.dot(swap_U, E))
    gammaU = math.dot(u, math.T(u))
    evs, _ = math.linalg.eig(gammaU)
    angles = [math.angle(ev) for ev in evs]
    if not qml.math.is_abstract(U):
        angles = math.sort(angles)
    x, y, z = (angles[0], angles[1], angles[2])
    alpha = (x + y) / 2
    beta = (x + z) / 2
    delta = (z + y) / 2
    interior_decomp = [qml.CNOT(wires=[wires[1], wires[0]]), qml.RZ(delta, wires=wires[0]), qml.RY(beta, wires=wires[1]), qml.CNOT(wires=wires), qml.RY(alpha, wires=wires[1]), qml.CNOT(wires=[wires[1], wires[0]])]
    RZd = qml.RZ(math.cast_like(delta, 1j), wires=wires[0]).matrix()
    RYb = qml.RY(beta, wires=wires[0]).matrix()
    RYa = qml.RY(alpha, wires=wires[0]).matrix()
    V_mats = [CNOT10, math.kron(RZd, RYb), CNOT01, math.kron(math.eye(2), RYa), CNOT10, SWAP]
    V = math.convert_like(math.eye(4), U)
    for mat in V_mats:
        V = math.dot(math.cast_like(mat, U), V)
    A, B, C, D = _extract_su2su2_prefactors(swap_U, V)
    A_ops = one_qubit_decomposition(A, wires[1])
    B_ops = one_qubit_decomposition(B, wires[0])
    C_ops = one_qubit_decomposition(C, wires[0])
    D_ops = one_qubit_decomposition(D, wires[1])
    return C_ops + D_ops + interior_decomp + A_ops + B_ops