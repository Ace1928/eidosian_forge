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
def activate_vdw_potential(self, dispersion_functional: str, potential_type: str) -> None:
    """
        Activate van der Waals dispersion corrections.

        Args:
            dispersion_functional: Type of dispersion functional.
                Options: pair_potential or non_local
            potential_type: What type of potential to use, given a dispersion functional type
                Options: DFTD2, DFTD3, DFTD3(BJ), DRSLL, LMKLL, RVV10
        """
    vdw = Section('VDW_POTENTIAL', keywords={'DISPERSION_FUNCTIONAL': Keyword('DISPERSION_FUNCTIONAL', dispersion_functional)})
    keywords = {'TYPE': Keyword('TYPE', potential_type)}
    if dispersion_functional.upper() == 'PAIR_POTENTIAL':
        reference_functional = self.xc_functionals[0]
        warnings.warn('Reference functional will not be checked for validity. Calculation will fail if the reference functional does not exist in the dftd3 reference data')
        keywords['PARAMETER_FILE_NAME'] = Keyword('PARAMETER_FILE_NAME', 'dftd3.dat')
        keywords['REFERENCE_FUNCTIONAL'] = Keyword('REFERENCE_FUNCTIONAL', reference_functional)
    vdw.insert(Section(dispersion_functional, keywords=keywords))
    self['FORCE_EVAL']['DFT']['XC'].insert(vdw)