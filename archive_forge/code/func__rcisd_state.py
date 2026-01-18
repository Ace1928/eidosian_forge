import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _rcisd_state(cisd_solver, tol=1e-15):
    """Construct a wavefunction from PySCF's ``RCISD`` solver object.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \\rangle` will be
    represented by the flipped binary string ``0011`` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \\rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    Args:
        cisd_solver (PySCF CISD Class instance): the class object representing the CISD calculation in PySCF
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form ``{(int_a, int_b) :coeff}``, with integers ``int_a, int_b``
        having binary representation corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> from pyscf import gto, scf, ci
    >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g', symmetry='d2h')
    >>> myhf = scf.RHF(mol).run()
    >>> myci = ci.CISD(myhf).run()
    >>> wf_cisd = _rcisd_state(myci, tol=1e-1)
    >>> print(wf_cisd)
    {(1, 1): -0.9942969785398775, (2, 2): 0.10664669927602162}
    """
    mol = cisd_solver.mol
    cisdvec = cisd_solver.ci
    norb = mol.nao
    nelec = mol.nelectron
    nocc, nvir = (nelec // 2, norb - nelec // 2)
    c0, c1, c2 = (cisdvec[0], cisdvec[1:nocc * nvir + 1], cisdvec[nocc * nvir + 1:].reshape(nocc, nocc, nvir, nvir))
    c2ab = c2.transpose(0, 2, 1, 3).reshape(nocc * nvir, -1)
    ref_a = int(2 ** nocc - 1)
    ref_b = ref_a
    fcimatr_dict = dict(zip(list(zip([ref_a], [ref_b])), [c0]))
    c1a_configs, c1a_signs = _excited_configurations(nocc, norb, 1)
    fcimatr_dict.update(dict(zip(list(zip(c1a_configs, [ref_b] * len(c1a_configs))), c1 * c1a_signs)))
    fcimatr_dict.update(dict(zip(list(zip([ref_a] * len(c1a_configs), c1a_configs)), c1 * c1a_signs)))
    if nocc > 1 and nvir > 1:
        c2_tr = c2 - c2.transpose(1, 0, 2, 3)
        ooidx, vvidx = (np.tril_indices(nocc, -1), np.tril_indices(nvir, -1))
        c2aa = c2_tr[ooidx][:, vvidx[0], vvidx[1]].ravel()
        c2aa_configs, c2aa_signs = _excited_configurations(nocc, norb, 2)
        fcimatr_dict.update(dict(zip(list(zip(c2aa_configs, [ref_b] * len(c2aa_configs))), c2aa * c2aa_signs)))
        fcimatr_dict.update(dict(zip(list(zip([ref_a] * len(c2aa_configs), c2aa_configs)), c2aa * c2aa_signs)))
    fcimatr_dict.update(dict(zip(list(product(c1a_configs, c1a_configs)), np.einsum('i,j,ij->ij', c1a_signs, c1a_signs, c2ab, optimize=True).ravel())))
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}
    fcimatr_dict = _sign_chem_to_phys(fcimatr_dict, norb)
    return fcimatr_dict