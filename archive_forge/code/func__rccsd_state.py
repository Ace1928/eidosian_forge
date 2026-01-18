import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _rccsd_state(ccsd_solver, tol=1e-15):
    """Construct a wavefunction from PySCF's ``RCCSD`` Solver object.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \\rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \\rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    In the current version, the exponential ansatz :math:`\\exp(\\hat{T}_1 + \\hat{T}_2) \\ket{\\text{HF}}`
    is expanded to second order, with only single and double excitation terms collected and kept.
    In the future this will be amended to also collect terms from higher order. The expansion gives

    .. math::
        \\exp(\\hat{T}_1 + \\hat{T}_2) \\ket{\\text{HF}} = \\left[ 1 + \\hat{T}_1 +
        \\left( \\hat{T}_2 + 0.5 * \\hat{T}_1^2 \\right) \\right] \\ket{\\text{HF}}

    The coefficients in this expansion are the CI coefficients used to build the wavefunction
    representation.

    Args:
        ccsd_solver (PySCF RCCSD Class instance): the class object representing the RCCSD calculation in PySCF
        tol (float): the tolerance for discarding Slater determinants with small coefficients

    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form ``{(int_a, int_b) :coeff}``, with integers ``int_a, int_b``
        having binary represention corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> from pyscf import gto, scf, cc
    >>> mol = gto.M(atom=[['Li', (0, 0, 0)], ['Li', (0,0,0.71)]], basis='sto6g', symmetry="d2h")
    >>> myhf = scf.RHF(mol).run()
    >>> mycc = cc.CCSD(myhf).run()
    >>> wf_ccsd = _rccsd_state(mycc, tol=1e-1)
    >>> print(wf_ccsd)
    {(7, 7): -0.8886969878256522, (11, 11): 0.30584590248164206,
     (19, 19): 0.30584590248164145, (35, 35): 0.14507552651170982}
    """
    mol = ccsd_solver.mol
    norb = mol.nao
    nelec = mol.nelectron
    nelec_a = int((nelec + mol.spin) / 2)
    nelec_b = int((nelec - mol.spin) / 2)
    nvir_a, nvir_b = (norb - nelec_a, norb - nelec_b)
    t1a = ccsd_solver.t1
    t1b = t1a
    t2aa = ccsd_solver.t2 - ccsd_solver.t2.transpose(1, 0, 2, 3)
    t2ab = ccsd_solver.t2.transpose(0, 2, 1, 3)
    t2bb = t2aa
    t2aa = t2aa - 0.5 * np.kron(t1a, t1a).reshape(nelec_a, nvir_a, nelec_a, nvir_a).transpose(0, 2, 1, 3)
    t2bb = t2bb - 0.5 * np.kron(t1b, t1b).reshape(nelec_b, nvir_b, nelec_b, nvir_b).transpose(0, 2, 1, 3)
    t2ab = t2ab - 0.5 * np.kron(t1a, t1b).reshape(nelec_a, nvir_a, nelec_b, nvir_b)
    ref_a = int(2 ** nelec_a - 1)
    ref_b = int(2 ** nelec_b - 1)
    fcimatr_dict = dict(zip(list(zip([ref_a], [ref_b])), [1.0]))
    t1a_configs, t1a_signs = _excited_configurations(nelec_a, norb, 1)
    fcimatr_dict.update(dict(zip(list(zip(t1a_configs, [ref_b] * len(t1a_configs))), t1a.ravel() * t1a_signs)))
    t1b_configs, t1b_signs = _excited_configurations(nelec_b, norb, 1)
    fcimatr_dict.update(dict(zip(list(zip([ref_a] * len(t1b_configs), t1b_configs)), t1b.ravel() * t1b_signs)))
    if nelec_a > 1 and nvir_a > 1:
        t2aa_configs, t2aa_signs = _excited_configurations(nelec_a, norb, 2)
        ooidx = np.tril_indices(nelec_a, -1)
        vvidx = np.tril_indices(nvir_a, -1)
        t2aa = t2aa[ooidx][:, vvidx[0], vvidx[1]]
        fcimatr_dict.update(dict(zip(list(zip(t2aa_configs, [ref_b] * len(t2aa_configs))), t2aa.ravel() * t2aa_signs)))
    if nelec_b > 1 and nvir_b > 1:
        t2bb_configs, t2bb_signs = _excited_configurations(nelec_b, norb, 2)
        ooidx = np.tril_indices(nelec_b, -1)
        vvidx = np.tril_indices(nvir_b, -1)
        t2bb = t2bb[ooidx][:, vvidx[0], vvidx[1]]
        fcimatr_dict.update(dict(zip(list(zip([ref_a] * len(t2bb_configs), t2bb_configs)), t2bb.ravel() * t2bb_signs)))
    fcimatr_dict.update(dict(zip(list(product(t1a_configs, t1b_configs)), np.einsum('i,j,ij->ij', t1a_signs, t1b_signs, t2ab.reshape(nelec_a * nvir_a, -1), optimize=True).ravel())))
    norm = np.sqrt(np.sum(np.array(list(fcimatr_dict.values())) ** 2))
    fcimatr_dict = {key: value / norm for key, value in fcimatr_dict.items()}
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}
    fcimatr_dict = _sign_chem_to_phys(fcimatr_dict, norb)
    return fcimatr_dict