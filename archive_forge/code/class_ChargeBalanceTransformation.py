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
class ChargeBalanceTransformation(AbstractTransformation):
    """This is a transformation that disorders a structure to make it charge
    balanced, given an oxidation state-decorated structure.
    """

    def __init__(self, charge_balance_sp):
        """
        Args:
            charge_balance_sp: specie to add or remove. Currently only removal
                is supported.
        """
        self.charge_balance_sp = str(charge_balance_sp)

    def apply_transformation(self, structure: Structure):
        """Applies the transformation.

        Args:
            structure: Input Structure

        Returns:
            Charge balanced structure.
        """
        charge = structure.charge
        specie = get_el_sp(self.charge_balance_sp)
        num_to_remove = charge / specie.oxi_state if specie.oxi_state else 0.0
        num_in_structure = structure.composition[specie]
        removal_fraction = num_to_remove / num_in_structure
        if removal_fraction < 0:
            raise ValueError('addition of specie not yet supported by ChargeBalanceTransformation')
        trans = SubstitutionTransformation({self.charge_balance_sp: {self.charge_balance_sp: 1 - removal_fraction}})
        return trans.apply_transformation(structure)

    def __repr__(self):
        return f'Charge Balance Transformation : Species to remove = {self.charge_balance_sp}'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False