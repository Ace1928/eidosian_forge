from __future__ import annotations
import logging
import math
import warnings
from fractions import Fraction
from itertools import groupby, product
from math import gcd
from string import ascii_lowercase
from typing import TYPE_CHECKING, Callable, Literal
import numpy as np
from joblib import Parallel, delayed
from monty.dev import requires
from monty.fractions import lcm
from monty.json import MSONable
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.energy_models import SymmetryModel
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.structure_matcher import SpinComparator, StructureMatcher
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.command_line.enumlib_caller import EnumError, EnumlibAdaptor
from pymatgen.command_line.mcsqs_caller import run_mcsqs
from pymatgen.core import DummySpecies, Element, Species, Structure, get_el_sp
from pymatgen.core.surface import SlabGenerator
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.icet import IcetSQS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
from pymatgen.transformations.transformation_abc import AbstractTransformation
def _add_spin_magnitudes(self, structure: Structure):
    """Replaces Spin.up/Spin.down with spin magnitudes specified by mag_species_spin.

        Args:
            structure (Structure): Structure to modify.

        Returns:
            Structure: Structure with spin magnitudes added.
        """
    for idx, site in enumerate(structure):
        if getattr(site.specie, 'spin', None):
            spin = site.specie.spin
            spin = getattr(site.specie, 'spin', None)
            sign = int(spin) if spin else 0
            if spin:
                sp = str(site.specie).split(',', maxsplit=1)[0]
                new_spin = sign * self.mag_species_spin.get(sp, 0)
                new_specie = Species(site.specie.symbol, getattr(site.specie, 'oxi_state', None), spin=new_spin)
                structure.replace(idx, new_specie, properties=site.properties)
    logger.debug(f'Structure with spin magnitudes:\n{structure}')
    return structure