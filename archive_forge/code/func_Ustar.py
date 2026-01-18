from __future__ import annotations
import re
from functools import partial
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifFile, CifParser, CifWriter, str2float
from pymatgen.symmetry.groups import SYMM_DATA
from pymatgen.util.due import Doi, due
@property
def Ustar(self) -> np.ndarray:
    """Computation as described in R. W. Grosse-Kunstleve, P. D. Adams, J Appl Cryst 2002, 35, 477-480.

        Returns:
            np.array: Ustar as array. First dimension are the atoms in the structure.
        """
    A = self.structure.lattice.matrix.T
    Ainv = np.linalg.inv(A)
    Ustar = []
    for mat in self.thermal_displacement_matrix_cart_matrixform:
        mat_ustar = np.dot(np.dot(Ainv, mat), Ainv.T)
        Ustar.append(mat_ustar)
    return np.array(Ustar)