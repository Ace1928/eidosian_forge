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
def activate_fast_minimization(self, on) -> None:
    """Method to modify the set to use fast SCF minimization."""
    if on:
        ot = OrbitalTransformation(minimizer='DIIS', preconditioner='FULL_ALL', algorithm='IRAC', linesearch='2PNT')
        self.update({'FORCE_EVAL': {'DFT': {'SCF': {'OT': ot}}}})