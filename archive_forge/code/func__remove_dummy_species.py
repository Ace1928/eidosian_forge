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
@staticmethod
def _remove_dummy_species(structure):
    """Structure with dummy species removed, but their corresponding spin properties
        merged with the original sites. Used after performing enumeration.
        """
    if not structure.is_ordered:
        raise RuntimeError('Something went wrong with enumeration.')
    sites_to_remove = []
    logger.debug(f'Dummy species structure:\n{structure}')
    for idx, site in enumerate(structure):
        if isinstance(site.specie, DummySpecies):
            sites_to_remove.append(idx)
            spin = getattr(site.specie, 'spin', 0)
            neighbors = structure.get_neighbors(site, 0.05, include_index=True)
            if len(neighbors) != 1:
                raise RuntimeError(f"This shouldn't happen, found neighbors={neighbors!r}")
            orig_site_idx = neighbors[0][2]
            orig_specie = structure[orig_site_idx].specie
            new_specie = Species(orig_specie.symbol, getattr(orig_specie, 'oxi_state', None), spin=spin)
            structure.replace(orig_site_idx, new_specie, properties=structure[orig_site_idx].properties)
    structure.remove_sites(sites_to_remove)
    logger.debug(f'Structure with dummy species removed:\n{structure}')
    return structure