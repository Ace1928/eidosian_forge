import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _sign_chem_to_phys(fcimatr_dict, norb):
    """Convert the dictionary-form wavefunction from chemist sign convention
    for ordering the creation operators by spin (i.e. all spin-up operators
    on the left) to the physicist convention native to PennyLane, which
    storing spin operators as interleaved for the same spatial orbital index.

    Note that convention change in the opposite direction -- starting from physicist
    and going to chemist -- can be accomplished with the same function
    (the sign transformation is reversible).

    Args:
        fcimatr_dict (dict[tuple(int, int), float]): dictionary of the form `{(int_a, int_b) :coeff}`, with integers `int_a, int_b`
        having binary represention corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations
        norb (int): total number of spatial orbitals of the underlying system

    Returns:
        signed_dict (dict): the same dictionary-type wavefunction with appropriate signs converted

    **Example**

    >>> fcimatr_dict = {(3, 1): 0.96, (6, 1): 0.1, \\
                        (3, 4): 0.1, (6, 4): 0.14, (5, 2): 0.19}
    >>> _sign_chem_to_phys(fcimatr_dict, 3)
    {(3, 1): -0.96, (6, 1): 0.1, (3, 4): 0.1, (6, 4): 0.14, (5, 2): -0.19}
    """
    signed_dict = {}
    for key, elem in fcimatr_dict.items():
        lsta, lstb = (bin(key[0])[2:][::-1], bin(key[1])[2:][::-1])
        lsta = np.array([int(elem) for elem in lsta] + [0] * (norb - len(lsta)))
        lstb = np.array([int(elem) for elem in lstb] + [0] * (norb - len(lstb)))
        which_occ = np.where(lsta == 1)[0]
        parity = (-1) ** np.sum([np.sum(lstb[:int(ind)]) for ind in which_occ])
        signed_dict[key] = parity * elem
    return signed_dict