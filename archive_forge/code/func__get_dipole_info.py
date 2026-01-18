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
def _get_dipole_info(filepath):
    """Internal command to process dipoles.

        Args:
            filepath (str): The path to the DDEC6_even_tempered_net_atomic_charges.xyz file
        """
    idx = 0
    start = False
    dipoles = []
    with open(filepath) as r:
        for line in r:
            if 'The following XYZ' in line:
                start = True
                idx += 1
                continue
            if start and line.strip() == '':
                break
            if idx >= 2:
                dipoles.append([float(d) for d in line.strip().split()[7:10]])
            if start:
                idx += 1
    return dipoles