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
def _write_jobscript_for_chargemol(self, net_charge=0.0, periodicity=(True, True, True), method='ddec6', compute_bond_orders=True):
    """Writes job_script.txt for Chargemol execution.

        Args:
            net_charge (float): Net charge of the system.
                Defaults to 0.0.
            periodicity (tuple[bool]): Periodicity of the system.
                Default: (True, True, True).
            method (str): Method to use for the analysis. Options include "ddec6"
                and "ddec3". Default: "ddec6"
            compute_bond_orders (bool): Whether to compute bond orders. Default: True.
        """
    self.net_charge = net_charge
    self.periodicity = periodicity
    self.method = method
    lines = ''
    if net_charge:
        lines += f'<net charge>\n{net_charge}\n</net charge>\n'
    if periodicity:
        per_a = '.true.' if periodicity[0] else '.false.'
        per_b = '.true.' if periodicity[1] else '.false.'
        per_c = '.true.' if periodicity[2] else '.false.'
        lines += f'<periodicity along A, B, and C vectors>\n{per_a}\n{per_b}\n{per_c}\n</periodicity along A, B, and C vectors>\n'
    atomic_densities_path = self._atomic_densities_path or os.getenv('DDEC6_ATOMIC_DENSITIES_DIR')
    if atomic_densities_path is None:
        raise OSError('The DDEC6_ATOMIC_DENSITIES_DIR environment variable must be set or the atomic_densities_path must be specified')
    if not os.path.isfile(atomic_densities_path):
        raise FileNotFoundError(f'atomic_densities_path={atomic_densities_path!r} does not exist')
    if os.name == 'nt':
        if atomic_densities_path[-1] != '\\':
            atomic_densities_path += '\\'
    elif atomic_densities_path[-1] != '/':
        atomic_densities_path += '/'
    lines += f'\n<atomic densities directory complete path>\n{atomic_densities_path}\n</atomic densities directory complete path>\n'
    lines += f'\n<charge type>\n{method.upper()}\n</charge type>\n'
    if compute_bond_orders:
        bo = '.true.' if compute_bond_orders else '.false.'
        lines += f'\n<compute BOs>\n{bo}\n</compute BOs>\n'
    with open('job_control.txt', mode='w') as file:
        file.write(lines)