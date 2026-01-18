from __future__ import annotations
import os
import subprocess
import warnings
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core import Element
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
@staticmethod
def _get_data_from_xyz(xyz_path):
    """Internal command to process Chargemol XYZ files.

        Args:
            xyz_path (str): Path to XYZ file

        Returns:
            list[float]: site-specific properties
        """
    props = []
    if os.path.isfile(xyz_path):
        with open(xyz_path) as r:
            for idx, line in enumerate(r):
                if idx <= 1:
                    continue
                if line.strip() == '':
                    break
                props.append(float(line.split()[-1]))
    else:
        raise FileNotFoundError(f'{xyz_path} not found')
    return props