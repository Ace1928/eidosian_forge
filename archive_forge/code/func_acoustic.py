from math import pi, sqrt
import warnings
from pathlib import Path
import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import ase
import ase.units as units
from ase.parallel import world
from ase.dft import monkhorst_pack
from ase.io.trajectory import Trajectory
from ase.utils.filecache import MultiFileJSONCache
def acoustic(self, C_N):
    """Restore acoustic sumrule on force constants."""
    natoms = len(self.indices)
    C_N_temp = C_N.copy()
    for C in C_N_temp:
        for a in range(natoms):
            for a_ in range(natoms):
                C_N[self.offset, 3 * a:3 * a + 3, 3 * a:3 * a + 3] -= C[3 * a:3 * a + 3, 3 * a_:3 * a_ + 3]