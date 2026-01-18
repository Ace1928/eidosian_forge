from __future__ import annotations
import copy
import linecache
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.pwmat.inputs import ACstrExtractor, AtomConfig, LineLocator
class DosSpin(MSONable):
    """Extract information of DOS from DOS_SPIN file:
    - DOS.totalspin, DOS.totalspin_projected
    - DOS.spinup, DOS.spinup_projected
    - DOS.spindown, DOS.spindown_projected
    """

    def __init__(self, filename: PathLike):
        self.filename: PathLike = filename
        self._labels, self._dos = self._parse()

    def _parse(self):
        """Parse the DOS_SPIN file to get name and values of partial DOS.

        Returns:
            labels (list[str]): The label of DOS, e.g. Total, Cr-3S, ...
            dos (np.array): Value of density of state.
        """
        labels: list[str] = []
        labels = linecache.getline(str(self.filename), 1).split()[1:]
        dos_str: str = ''
        with zopen(self.filename, mode='rt') as file:
            file.readline()
            dos_str = file.read()
        dos: np.array = np.loadtxt(StringIO(dos_str))
        return (labels, dos)

    @property
    def labels(self) -> list[str]:
        """Returns the name of the partial density of states"""
        return self._labels

    @property
    def dos(self) -> np.ndarray:
        """Returns value of density of state."""
        return self._dos

    def get_partial_dos(self, part: str) -> np.ndarray:
        """Get partial dos for give element or orbital.

        Args:
            part (str): The name of partial dos.
                e.g. 'Energy', 'Total', 'Cr-3S', 'Cr-3P',
                    'Cr-4S', 'Cr-3D', 'I-4D', 'I-5S', 'I-5P', 'Cr-3S', 'Cr-3Pz',
                    'Cr-3Px', 'Cr-3Py', 'Cr-4S', 'Cr-3Dz2','Cr-3Dxz', 'Cr-3Dyz',
                    'Cr-3D(x^2-y^2)', 'Cr-3Dxy', 'I-4Dz2', 'I-4Dxz', 'I-4Dyz',
                    'I-4D(x^2-y^2)', 'I-4Dxy', 'I-5S', 'I-5Pz', 'I-5Px', 'I-5Py'

        Returns:
            partial_dos: np.array
        """
        part_upper: str = part.upper()
        labels_upper: list[str] = [tmp_label.upper() for tmp_label in self._labels]
        idx_dos = labels_upper.index(part_upper)
        return self._dos[:, idx_dos]