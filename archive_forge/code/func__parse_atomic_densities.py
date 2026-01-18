from __future__ import annotations
import os
import shutil
import subprocess
import warnings
from datetime import datetime
from glob import glob
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.dev import deprecated
from monty.shutil import decompress_file
from monty.tempfile import ScratchDir
from pymatgen.io.common import VolumetricData
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
@deprecated(message='parse_atomic_densities was deprecated on 2024-02-26 and will be removed on 2025-02-26.\nSee https://github.com/materialsproject/pymatgen/issues/3652 for details.')
def _parse_atomic_densities(self) -> list[dict]:
    """Parse atom-centered charge densities with excess zeros removed.

        Each dictionary has the keys:
            "data", "shift", "dim", where "data" is the charge density array,
            "shift" is the shift used to center the atomic charge density, and
            "dim" is the dimension of the original charge density.
        """
    if datetime(2025, 2, 26) < datetime.now() and os.getenv('CI') and (os.getenv('GITHUB_REPOSITORY') == 'materialsproject/pymatgen'):
        raise RuntimeError('This method should have been removed, see #3656.')

    def slice_from_center(data: np.ndarray, x_width: int, y_width: int, z_width: int) -> np.ndarray:
        """Slices a central window from the data array."""
        x, y, z = data.shape
        start_x = x // 2 - x_width // 2
        start_y = y // 2 - y_width // 2
        start_z = z // 2 - z_width // 2
        return data[start_x:start_x + x_width, start_y:start_y + y_width, start_z:start_z + z_width]

    def find_encompassing_vol(data: np.ndarray) -> np.ndarray | None:
        """Find the central encompassing volume which
            holds all the data within a precision.
            """
        total = np.sum(data)
        for idx in range(np.max(data.shape)):
            sliced_data = slice_from_center(data, idx, idx, idx)
            if total - np.sum(sliced_data) < 0.1:
                return sliced_data
        return None
    atom_chgcars = [Chgcar.from_file(f'BvAt{idx + 1:04}.dat') for idx in range(len(self.chgcar.structure))]
    atomic_densities = []
    for _site, loc, chg in zip(self.chgcar.structure, self.chgcar.structure.frac_coords, atom_chgcars):
        index = np.round(np.multiply(loc, chg.dim))
        shift = (np.divide(chg.dim, 2) - index).astype(int)
        shifted_data = np.roll(chg.data['total'], shift, axis=(0, 1, 2))
        dct = {'data': find_encompassing_vol(shifted_data), 'shift': shift, 'dim': self.chgcar.dim}
        atomic_densities.append(dct)
    return atomic_densities