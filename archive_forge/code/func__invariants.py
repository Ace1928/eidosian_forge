import numpy as np
import ase.units as u
from ase.parallel import world
from ase.phonons import Phonons
from ase.vibrations.vibrations import Vibrations, AtomicDisplacements
from ase.dft import monkhorst_pack
from ase.utils import IOContext
def _invariants(self, alpha_Qcc):
    """Raman invariants

        Parameter
        ---------
        alpha_Qcc: array
           Matrix element or polarizability tensor

        Reference
        ---------
        Derek A. Long, The Raman Effect, ISBN 0-471-49028-8

        Returns
        -------
        mean polarizability, anisotropy, asymmetric anisotropy
        """
    m2 = Raman.m2
    alpha2_r = m2(alpha_Qcc[:, 0, 0] + alpha_Qcc[:, 1, 1] + alpha_Qcc[:, 2, 2]) / 9.0
    delta2_r = 3 / 4.0 * (m2(alpha_Qcc[:, 0, 1] - alpha_Qcc[:, 1, 0]) + m2(alpha_Qcc[:, 0, 2] - alpha_Qcc[:, 2, 0]) + m2(alpha_Qcc[:, 1, 2] - alpha_Qcc[:, 2, 1]))
    gamma2_r = 3 / 4.0 * (m2(alpha_Qcc[:, 0, 1] + alpha_Qcc[:, 1, 0]) + m2(alpha_Qcc[:, 0, 2] + alpha_Qcc[:, 2, 0]) + m2(alpha_Qcc[:, 1, 2] + alpha_Qcc[:, 2, 1])) + (m2(alpha_Qcc[:, 0, 0] - alpha_Qcc[:, 1, 1]) + m2(alpha_Qcc[:, 0, 0] - alpha_Qcc[:, 2, 2]) + m2(alpha_Qcc[:, 1, 1] - alpha_Qcc[:, 2, 2])) / 2
    return (alpha2_r, gamma2_r, delta2_r)