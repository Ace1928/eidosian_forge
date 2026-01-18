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
def activate_nonperiodic(self, solver='ANALYTIC') -> None:
    """
        Activates a calculation with non-periodic calculations by turning of PBC and
        changing the poisson solver. Still requires a CELL to put the atoms.
        """
    kwds = {'POISSON_SOLVER': Keyword('POISSON_SOLVER', solver), 'PERIODIC': Keyword('PERIODIC', 'NONE')}
    self['FORCE_EVAL']['DFT'].insert(Section('POISSON', subsections={}, keywords=kwds))