from __future__ import annotations
import itertools
import os
import warnings
import numpy as np
from ruamel.yaml import YAML
from pymatgen.core import SETTINGS
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Element, Molecule, Structure
from pymatgen.io.cp2k.inputs import (
from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff, get_unique_site_indices
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
def activate_robust_minimization(self) -> None:
    """Method to modify the set to use more robust SCF minimization technique."""
    ot = OrbitalTransformation(minimizer='CG', preconditioner='FULL_ALL', algorithm='STRICT', linesearch='3PNT')
    self.update({'FORCE_EVAL': {'DFT': {'SCF': {'OT': ot}}}})