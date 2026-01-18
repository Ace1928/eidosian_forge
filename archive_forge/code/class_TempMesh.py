from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.dev import requires
from monty.json import MSONable
from scipy.interpolate import UnivariateSpline
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import amu_to_kg
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
class TempMesh:
    """Temporary Class."""