import numpy as np
import pennylane as qml
from pennylane import math
from .single_qubit_unitary import one_qubit_decomposition
def _decomposition_2_cnots(U, wires):
    """If 2 CNOTs are required, we can write the circuit as
     -╭U- = -A--╭X--RZ(d)--╭X--C-
     -╰U- = -B--╰C--RX(p)--╰C--D-
    We need to find the angles for the Z and X rotations such that the inner
    part has the same spectrum as U, and then we can recover A, B, C, D.
    """
    u = math.dot(Edag, math.dot(U, E))
    gammaU = math.dot(u, math.T(u))
    evs, _ = math.linalg.eig(gammaU)
    sorted_evs = math.sort(math.real(evs))
    if math.allclose(sorted_evs, [-1, -1, 1, 1]):
        interior_decomp = [qml.CNOT(wires=[wires[1], wires[0]]), qml.S(wires=wires[0]), qml.SX(wires=wires[1]), qml.CNOT(wires=[wires[1], wires[0]])]
        inner_matrix = S_SX
    else:
        x = math.angle(evs[0])
        y = math.angle(evs[1])
        if math.allclose(x, -y):
            y = math.angle(evs[2])
        delta = (x + y) / 2
        phi = (x - y) / 2
        interior_decomp = [qml.CNOT(wires=[wires[1], wires[0]]), qml.RZ(delta, wires=wires[0]), qml.RX(phi, wires=wires[1]), qml.CNOT(wires=[wires[1], wires[0]])]
        RZd = qml.RZ(math.cast_like(delta, 1j), wires=0).matrix()
        RXp = qml.RX(phi, wires=0).matrix()
        inner_matrix = math.kron(RZd, RXp)
    V = math.dot(math.cast_like(CNOT10, U), math.dot(inner_matrix, math.cast_like(CNOT10, U)))
    A, B, C, D = _extract_su2su2_prefactors(U, V)
    A_ops = one_qubit_decomposition(A, wires[0])
    B_ops = one_qubit_decomposition(B, wires[1])
    C_ops = one_qubit_decomposition(C, wires[0])
    D_ops = one_qubit_decomposition(D, wires[1])
    return C_ops + D_ops + interior_decomp + A_ops + B_ops