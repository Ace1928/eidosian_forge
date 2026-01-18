from __future__ import annotations
import copy
import linecache
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.pwmat.inputs import ACstrExtractor, AtomConfig, LineLocator
def _parse_kpt(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Parse REPORT file to obtain information about kpoints.

        Returns:
            3-tuple containing:
                kpts (np.ndarray): The fractional coordinates of kpoints.
                kpts_weight (np.ndarray): The weight of kpoints.
                hsps (dict[str, np.ndarray]): The name and coordinates of high symmetric points.
        """
    num_rows: int = int(self._num_kpts)
    content: str = 'total number of K-point:'
    row_idx: int = LineLocator.locate_all_lines(self.filename, content)[0]
    kpts: np.array = np.zeros((self._num_kpts, 3))
    kpts_weight: np.array = np.zeros(self._num_kpts)
    hsps: dict[str, np.array] = {}
    for ii in range(num_rows):
        tmp_row_lst: list[str] = linecache.getline(str(self.filename), row_idx + ii + 1).split()
        for jj in range(3):
            kpts[ii][jj] = float(tmp_row_lst[jj].strip())
        kpts_weight[ii] = float(tmp_row_lst[3].strip())
        if len(tmp_row_lst) == 5:
            hsps.update({tmp_row_lst[4]: np.array([float(tmp_row_lst[0]), float(tmp_row_lst[1]), float(tmp_row_lst[2])])})
    return (kpts, kpts_weight, hsps)