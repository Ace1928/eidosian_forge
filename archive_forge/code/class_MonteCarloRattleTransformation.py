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
class MonteCarloRattleTransformation(AbstractTransformation):
    """Uses a Monte Carlo rattle procedure to randomly perturb the sites in a
    structure.

    This class requires the hiPhive package to be installed.

    Rattling atom `i` is carried out as a Monte Carlo move that is accepted with
    a probability determined from the minimum interatomic distance
    :math:`d_{ij}`. If :math:`\\\\min(d_{ij})` is smaller than :math:`d_{min}`
    the move is only accepted with a low probability.

    This process is repeated for each atom a number of times meaning
    the magnitude of the final displacements is not *directly*
    connected to `rattle_std`.
    """

    @requires(hiphive, 'hiphive is required for MonteCarloRattleTransformation')
    def __init__(self, rattle_std: float, min_distance: float, seed: int | None=None, **kwargs):
        """
        Args:
            rattle_std: Rattle amplitude (standard deviation in normal
                distribution). Note: this value is not *directly* connected to the
                final average displacement for the structures
            min_distance: Interatomic distance used for computing the probability
                for each rattle move.
            seed: Seed for setting up NumPy random state from which random numbers
                are generated. If ``None``, a random seed will be generated
                (default). This option allows the output of this transformation
                to be deterministic.
            **kwargs: Additional keyword arguments to be passed to the hiPhive
                mc_rattle function.
        """
        self.rattle_std = rattle_std
        self.min_distance = min_distance
        self.seed = seed
        if not seed:
            seed = np.random.randint(1, 1000000000)
        self.random_state = np.random.RandomState(seed)
        self.kwargs = kwargs

    def apply_transformation(self, structure: Structure) -> Structure:
        """Apply the transformation.

        Args:
            structure: Input Structure

        Returns:
            Structure with sites perturbed.
        """
        from hiphive.structure_generation.rattle import mc_rattle
        atoms = AseAtomsAdaptor.get_atoms(structure)
        seed = self.random_state.randint(1, 1000000000)
        displacements = mc_rattle(atoms, self.rattle_std, self.min_distance, seed=seed, **self.kwargs)
        return Structure(structure.lattice, structure.species, structure.cart_coords + displacements, coords_are_cartesian=True)

    def __repr__(self):
        return f'{__name__} : rattle_std = {self.rattle_std}'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False