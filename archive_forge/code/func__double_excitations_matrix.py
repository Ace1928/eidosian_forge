from quantum chemistry applications.
import functools
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import Operation
def _double_excitations_matrix(phi, phase_prefactor):
    """This helper function unifies the `compute_matrix` methods
    of `DoubleExcitation`, `DoubleExcitationPlus` and `DoubleExcitationMinus`.
    `phase_prefactor` determines which operation is produced:
        `phase_prefactor=0.` : `DoubleExcitation`
        `phase_prefactor=0.5j` : `DoubleExcitationPlus`
        `phase_prefactor=-0.5j` : `DoubleExcitationMinus`
    """
    interface = qml.math.get_interface(phi)
    if interface == 'tensorflow' and isinstance(phase_prefactor, complex):
        phi = qml.math.cast_like(phi, 1j)
    c = qml.math.cos(phi / 2)
    s = qml.math.sin(phi / 2)
    e = qml.math.exp(phase_prefactor * phi)
    if qml.math.ndim(phi) == 0:
        diag = qml.math.diag([e] * 3 + [c] + [e] * 8 + [c] + [e] * 3)
        if interface == 'torch':
            return diag + s * qml.math.convert_like(DoubleExcitation.mask_s, phi)
        return diag + s * DoubleExcitation.mask_s
    if isinstance(phase_prefactor, complex):
        c = (1 + 0j) * c
    diag = qml.math.stack([e] * 3 + [c] + [e] * 8 + [c] + [e] * 3, axis=-1)
    diag = qml.math.einsum('ij,jk->ijk', diag, I16)
    off_diag = qml.math.einsum('i,jk->ijk', s, DoubleExcitation.mask_s)
    return diag + off_diag