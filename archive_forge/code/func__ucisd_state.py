import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _ucisd_state(cisd_solver, tol=1e-15):
    """Construct a wavefunction from PySCF's ``UCISD`` solver object.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \\rangle` will be
    represented by the flipped binary string ``0011`` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with `0.99` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \\rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    Args:
        cisd_solver (PySCF UCISD Class instance): the class object representing the UCISD calculation in PySCF
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form ``{(int_a, int_b) :coeff}``, with integers ``int_a, int_b``
        having binary representation corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> from pyscf import gto, scf, ci
    >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g', symmetry='d2h')
    >>> myhf = scf.UHF(mol).run()
    >>> myci = ci.UCISD(myhf).run()
    >>> wf_cisd = _ucisd_state(myci, tol=1e-1)
    >>> print(wf_cisd)
    {(1, 1): -0.9942969785398778, (2, 2): 0.10664669927602159}
    """
    mol = cisd_solver.mol
    cisdvec = cisd_solver.ci
    norb = mol.nao
    nelec = mol.nelectron
    nelec_a = int((nelec + mol.spin) / 2)
    nelec_b = int((nelec - mol.spin) / 2)
    nvir_a, nvir_b = (norb - nelec_a, norb - nelec_b)
    size_a, size_b = (nelec_a * nvir_a, nelec_b * nvir_b)
    size_aa = int(nelec_a * (nelec_a - 1) / 2) * int(nvir_a * (nvir_a - 1) / 2)
    size_bb = int(nelec_b * (nelec_b - 1) / 2) * int(nvir_b * (nvir_b - 1) / 2)
    size_ab = size_a * size_b
    cumul = np.cumsum([0, 1, size_a, size_b, size_ab, size_aa, size_bb])
    c0, c1a, c1b, c2ab, c2aa, c2bb = [cisdvec[cumul[idx]:cumul[idx + 1]] for idx in range(len(cumul) - 1)]
    c2ab = c2ab.reshape(nelec_a, nelec_b, nvir_a, nvir_b).transpose(0, 2, 1, 3).reshape(nelec_a * nvir_a, -1)
    ref_a = int(2 ** nelec_a - 1)
    ref_b = int(2 ** nelec_b - 1)
    fcimatr_dict = dict(zip(list(zip([ref_a], [ref_b])), c0))
    c1a_configs, c1a_signs = _excited_configurations(nelec_a, norb, 1)
    fcimatr_dict.update(dict(zip(list(zip(c1a_configs, [ref_b] * size_a)), c1a * c1a_signs)))
    c1b_configs, c1b_signs = _excited_configurations(nelec_b, norb, 1)
    fcimatr_dict.update(dict(zip(list(zip([ref_a] * size_b, c1b_configs)), c1b * c1b_signs)))
    c2aa_configs, c2aa_signs = _excited_configurations(nelec_a, norb, 2)
    fcimatr_dict.update(dict(zip(list(zip(c2aa_configs, [ref_b] * size_aa)), c2aa * c2aa_signs)))
    fcimatr_dict.update(dict(zip(list(product(c1a_configs, c1b_configs)), np.einsum('i,j,ij->ij', c1a_signs, c1b_signs, c2ab, optimize=True).ravel())))
    c2bb_configs, c2bb_signs = _excited_configurations(nelec_b, norb, 2)
    fcimatr_dict.update(dict(zip(list(zip([ref_a] * size_bb, c2bb_configs)), c2bb * c2bb_signs)))
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}
    fcimatr_dict = _sign_chem_to_phys(fcimatr_dict, norb)
    return fcimatr_dict