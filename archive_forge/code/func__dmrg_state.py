import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _dmrg_state(wavefunction, tol=1e-15):
    """Construct a wavefunction from the DMRG wavefunction obtained from the Block2 library.

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

    The determinants and coefficients should be supplied externally. They should be calculated by
    using Block2 DMRGDriver's `get_csf_coefficients()` method.

    Args:
        wavefunction tuple(list[int], array[float]): determinants and coefficients in physicist notation
        tol (float): the tolerance for discarding Slater determinants with small coefficients

    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form `{(int_a, int_b) : coeff}`, with integers `int_a, int_b`
        having binary representation corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> import numpy as np
    >>> wavefunction = ( [[0, 3], [3, 0]], np.array([-0.10660077,  0.9943019 ]))
    >>> wf_dmrg = _dmrg_state(wavefunction, tol=1e-1)
    >>> print(wf_dmrg)
    {(2, 2): -0.10660077, (1, 1): 0.9943019}
    """
    dets, coeffs = wavefunction
    row, col = ([], [])
    for det in dets:
        stra, strb = _sitevec_to_fock(det, format='dmrg')
        row.append(stra)
        col.append(strb)
    fcimatr_dict = dict(zip(list(zip(row, col)), coeffs))
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}
    return fcimatr_dict