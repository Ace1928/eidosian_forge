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
class MultipleSubstitutionTransformation:
    """Performs multiple substitutions on a structure. For example, can do a
    fractional replacement of Ge in LiGePS with a list of species, creating one
    structure for each substitution. Ordering is done using a dummy element so
    only one ordering must be done per substitution oxidation state. Charge
    balancing of the structure is optionally performed.

    Note:
        There are no checks to make sure that removal fractions are possible and rounding
        may occur. Currently charge balancing only works for removal of species.
    """

    def __init__(self, sp_to_replace, r_fraction, substitution_dict, charge_balance_species=None, order=True):
        """Performs multiple fractional substitutions on a transmuter.

        Args:
            sp_to_replace: species to be replaced
            r_fraction: fraction of that specie to replace
            substitution_dict: dictionary of the format
                {2: ["Mg", "Ti", "V", "As", "Cr", "Ta", "N", "Nb"],
                3: ["Ru", "Fe", "Co", "Ce", "As", "Cr", "Ta", "N", "Nb"],
                4: ["Ru", "V", "Cr", "Ta", "N", "Nb"],
                5: ["Ru", "W", "Mn"]
                }
                The number is the charge used for each of the list of elements
                (an element can be present in multiple lists)
            charge_balance_species: If specified, will balance the charge on
                the structure using that specie.
            order: Whether to order the structures.
        """
        self.sp_to_replace = sp_to_replace
        self.r_fraction = r_fraction
        self.substitution_dict = substitution_dict
        self.charge_balance_species = charge_balance_species
        self.order = order

    def apply_transformation(self, structure: Structure, return_ranked_list: bool | int=False):
        """Applies the transformation.

        Args:
            structure: Input Structure
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures

                is returned. If False, only the single lowest energy structure is returned. Defaults to False.

        Returns:
            Structures with all substitutions applied.
        """
        if not return_ranked_list:
            raise ValueError('MultipleSubstitutionTransformation has no single best structure output. Must use return_ranked_list.')
        outputs = []
        for charge, el_list in self.substitution_dict.items():
            sign = '+' if charge > 0 else '-'
            dummy_sp = f'X{charge}{sign}'
            mapping = {self.sp_to_replace: {self.sp_to_replace: 1 - self.r_fraction, dummy_sp: self.r_fraction}}
            trans = SubstitutionTransformation(mapping)
            dummy_structure = trans.apply_transformation(structure)
            if self.charge_balance_species is not None:
                cbt = ChargeBalanceTransformation(self.charge_balance_species)
                dummy_structure = cbt.apply_transformation(dummy_structure)
            if self.order:
                trans = OrderDisorderedStructureTransformation()
                dummy_structure = trans.apply_transformation(dummy_structure)
            for el in el_list:
                sign = '+' if charge > 0 else '-'
                st = SubstitutionTransformation({f'X{charge}+': f'{el}{charge}{sign}'})
                new_structure = st.apply_transformation(dummy_structure)
                outputs.append({'structure': new_structure})
        return outputs

    def __repr__(self):
        return f'Multiple Substitution Transformation : Substitution on {self.sp_to_replace}'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: True."""
        return True