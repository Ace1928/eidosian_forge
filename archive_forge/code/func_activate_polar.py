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
def activate_polar(self, **kwargs) -> None:
    """Calculate polarizations (including raman)."""
    if not self.check('force_eval/properties'):
        self['FORCE_EVAL'].insert(Section('PROPERTIES'))
    if not self.check('force_eval/properties/linres'):
        self['FORCE_EVAL']['PROPERTIES'].insert('LINRES')
    self['FORCE_EVAL']['PROPERTIES']['LINRES'].insert(Section('POLAR', **kwargs))