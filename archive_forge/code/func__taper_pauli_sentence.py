import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def _taper_pauli_sentence(ps_h, generators, paulixops, paulix_sector):
    """Transform a PauliSentence with a Clifford operator and then taper qubits.

    Args:
        ps_h (~.PauliSentence): The Hamiltonian to be tapered
        generators (list[Operator]): generators expressed as PennyLane Hamiltonians
        paulixops (list[~.PauliX]): list of single-qubit Pauli-X operators
        paulix_sector (list[int]): eigenvalues of the Pauli-X operators.

    Returns:
        (Operator): the tapered Hamiltonian
    """
    u = clifford(generators, paulixops)
    ps_u = pauli_sentence(u)
    ts_ps = qml.pauli.PauliSentence()
    for ps in _split_pauli_sentence(ps_h, max_size=PAULI_SENTENCE_MEMORY_SPLITTING_SIZE):
        ts_ps += ps_u @ ps @ ps_u
    wireset = ps_u.wires + ps_h.wires
    wiremap = dict(zip(list(wireset.toset()), range(len(wireset) + 1)))
    paulix_wires = [x.wires[0] for x in paulixops]
    o = []
    val = np.ones(len(ts_ps))
    wires_tap = [i for i in ts_ps.wires if i not in paulix_wires]
    wiremap_tap = dict(zip(wires_tap, range(len(wires_tap) + 1)))
    for i, pw_coeff in enumerate(ts_ps.items()):
        pw, _ = pw_coeff
        for idx, w in enumerate(paulix_wires):
            if pw[w] == 'X':
                val[i] *= paulix_sector[idx]
        o.append(qml.pauli.string_to_pauli_word(''.join([pw[wiremap[i]] for i in wires_tap]), wire_map=wiremap_tap))
    c = qml.math.stack(qml.math.multiply(val * complex(1.0), list(ts_ps.values())))
    tapered_ham = qml.simplify(qml.dot(c, o)) if active_new_opmath() else simplify(qml.Hamiltonian(c, o))
    if set(wires_tap) != tapered_ham.wires.toset():
        identity_op = functools.reduce(lambda i, j: i @ j, [qml.Identity(wire) for wire in Wires.unique_wires([tapered_ham.wires, Wires(wires_tap)])])
        if active_new_opmath():
            return tapered_ham + 0.0 * identity_op
        tapered_ham = qml.Hamiltonian(np.array([*tapered_ham.coeffs, 0.0]), [*tapered_ham.ops, identity_op])
    return tapered_ham