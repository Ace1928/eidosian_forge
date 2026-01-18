import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _excited_configurations(electrons, orbitals, excitation):
    """Generate excited configurations from a Hartree-Fock reference state.

    This function generates excited configurations in the form of integers representing a binary
    string, e.g., :math:`|1 1 0 1 0 0 \\rangle` is represented by :math:`int('110100', 2) = 52`.

    The excited configurations are generated from a Hartree-Fock (HF) reference state. The HF state
    is assumed to have the form :math:`|1 1 ...1 0 ... 0 0 \\rangle` where the number of :math:`1`
    and :math:`0` elements are the number of occupied and unoccupied spin orbitals, respectively.
    The string representation of the state is obtained by converting the occupation-number vector to
    a string, e.g., ``111000`` to represent :math:`|1 1 1 0 0 0 \\rangle.

    Each excited configuration has a sign, :math:`+1` or :math:`-1`, that is obtained by reordering
    the creation operators.

    Args:
        electrons (int): number of electrons
        orbitals (int): number of spin orbitals
        excitation (int): number of excited electrons

    Returns:
        tuple(list, list): lists of excited configurations and signs obtained by reordering the
         creation operators

    **Example**

    >>> electrons = 3
    >>> orbitals = 5
    >>> excitation = 2
    >>> _excited_configurations(electrons, orbitals, excitation)
    ([28, 26, 25], [1, -1, 1])
    """
    if excitation not in [1, 2]:
        raise ValueError('Only single (excitation = 1) and double (excitation = 2) excitations are supported.')
    hf_state = qml.qchem.hf_state(electrons, orbitals)
    singles, doubles = _excitations(electrons, orbitals)
    states, signs = ([], [])
    if excitation == 1:
        for s in singles:
            state = hf_state.copy()
            state[s] = state[[s[1], s[0]]]
            states += [state]
            signs.append((-1) ** len(range(s[0], electrons - 1)))
    if excitation == 2:
        for d in doubles:
            state = hf_state.copy()
            state[d] = state[[d[2], d[3], d[0], d[1]]]
            states += [state]
            order_pq = len(range(d[0], electrons - 1))
            order_rs = len(range(d[1], electrons - 1))
            signs.append((-1) ** (order_pq + order_rs + 1))
    states_str = [''.join([str(i) for i in state]) for state in states]
    states_int = [int(state[::-1], 2) for state in states_str]
    return (states_int, signs)