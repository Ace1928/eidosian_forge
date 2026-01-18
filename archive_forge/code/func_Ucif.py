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
def Ucif(self) -> np.ndarray:
    """Computation as described in R. W. Grosse-Kunstleve, P. D. Adams, J Appl Cryst 2002, 35, 477-480.

        Returns:
            np.array: Ucif as array. First dimension are the atoms in the structure.
        """
    if self.thermal_displacement_matrix_cif is None:
        A = self.structure.lattice.matrix.T
        N = np.diag([np.linalg.norm(x) for x in np.linalg.inv(A)])
        Ninv = np.linalg.inv(N)
        Ucif = []
        Ustar = self.Ustar
        for mat in Ustar:
            mat_cif = np.dot(np.dot(Ninv, mat), Ninv.T)
            Ucif.append(mat_cif)
        return np.array(Ucif)
    return self.thermal_displacement_matrix_cif_matrixform