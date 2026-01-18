import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _shci_state(wavefunction, tol=1e-15):
    """Construct a wavefunction from the SHCI wavefunction obtained from the Dice library.

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

    The determinants and coefficients should be supplied externally. They are typically stored under
    SHCI.outputfile.

    Args:
        wavefunction tuple(list[str], array[float]): determinants and coefficients in chemist notation
        tol (float): the tolerance for discarding Slater determinants with small coefficients
    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form `{(int_a, int_b) : coeff}`, with integers `int_a, int_b`
        having binary representation corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> import numpy as np
    >>> wavefunction = ( ['20', '02'], np.array([-0.9943019036, 0.1066007711]))
    >>> wf_shci = _shci_state(wavefunction, tol=1e-1)
    >>> print(wf_shci)
    {(1, 1): -0.9943019036, (2, 2): 0.1066007711}
    """
    dets, coeffs = wavefunction
    norb = len(dets[0])
    xa = []
    xb = []
    for det in dets:
        bin_a, bin_b = _sitevec_to_fock(list(det), 'shci')
        xa.append(bin_a)
        xb.append(bin_b)
    fcimatr_dict = dict(zip(list(zip(xa, xb)), coeffs))
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}
    fcimatr_dict = _sign_chem_to_phys(fcimatr_dict, norb)
    return fcimatr_dict