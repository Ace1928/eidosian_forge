from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def _auto_set_energy_range(self) -> None:
    """Automatically determine the energy range as min/max eigenvalue
        minus/plus the buffer_in_ev.
        """
    emins = [min((e_k[0] for e_k in self._bs.bands[Spin.up]))]
    emaxs = [max((e_k[0] for e_k in self._bs.bands[Spin.up]))]
    if self._bs.is_spin_polarized:
        emins.append(min((e_k[0] for e_k in self._bs.bands[Spin.down])))
        emaxs.append(max((e_k[0] for e_k in self._bs.bands[Spin.down])))
    min_eigenval = Energy(min(emins) - self._bs.efermi, 'eV').to('Ry')
    max_eigenval = Energy(max(emaxs) - self._bs.efermi, 'eV').to('Ry')
    const = Energy(2, 'eV').to('Ry')
    self._ll = min_eigenval - const
    self._hl = max_eigenval + const
    en_range = Energy(max((abs(self._ll), abs(self._hl))), 'Ry').to('eV')
    self.energy_span_around_fermi = en_range * 1.01
    print('energy_span_around_fermi = ', self.energy_span_around_fermi)