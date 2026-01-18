from __future__ import annotations
import copy
import linecache
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.pwmat.inputs import ACstrExtractor, AtomConfig, LineLocator
class OutFermi(MSONable):
    """Extract fermi energy (eV) from OUT.FERMI"""

    def __init__(self, filename: PathLike):
        """Initialization function

        Args:
            filename (PathLike): The absolute path of OUT.FERMI file.
        """
        self.filename: PathLike = filename
        with zopen(self.filename, 'rt') as file:
            self._e_fermi: float = np.round(float(file.readline().split()[-2].strip()), 3)

    @property
    def e_fermi(self) -> float:
        """Returns the fermi energy level.

        Returns:
            float: Fermi energy level.
        """
        return self._e_fermi