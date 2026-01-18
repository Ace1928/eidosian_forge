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
class SuperTransformation(AbstractTransformation):
    """This is a transformation that is inherently one-to-many. It is constructed
    from a list of transformations and returns one structure for each
    transformation. The primary use for this class is extending a transmuter
    object.
    """

    def __init__(self, transformations, nstructures_per_trans=1):
        """
        Args:
            transformations ([transformations]): List of transformations to apply
                to a structure. One transformation is applied to each output
                structure.
            nstructures_per_trans (int): If the transformations are one-to-many and,
                nstructures_per_trans structures from each transformation are
                added to the full list. Defaults to 1, i.e., only best structure.
        """
        self._transformations = transformations
        self.nstructures_per_trans = nstructures_per_trans

    def apply_transformation(self, structure: Structure, return_ranked_list: bool | int=False):
        """Applies the transformation.

        Args:
            structure: Input Structure
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures

                is returned. If False, only the single lowest energy structure is returned. Defaults to False.

        Returns:
            Structures with all transformations applied.
        """
        if not return_ranked_list:
            raise ValueError('SuperTransformation has no single best structure output. Must use return_ranked_list')
        structures = []
        for t in self._transformations:
            if t.is_one_to_many:
                for d in t.apply_transformation(structure, return_ranked_list=self.nstructures_per_trans):
                    d['transformation'] = t
                    structures.append(d)
            else:
                structures.append({'transformation': t, 'structure': t.apply_transformation(structure)})
        return structures

    def __repr__(self):
        return f'Super Transformation : Transformations = {' '.join(map(str, self._transformations))}'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: True."""
        return True