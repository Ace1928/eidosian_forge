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
def _sqs_cluster_estimate(struct_disordered, cluster_size_and_shell: dict[int, int] | None=None):
    """Set up an ATAT cluster.out file for a given structure and set of constraints.

        Args:
            struct_disordered: disordered pymatgen Structure object
            cluster_size_and_shell: dict of integers {cluster: shell}.

        Returns:
            dict of {cluster size: distance in angstroms} for mcsqs calculation
        """
    cluster_size_and_shell = cluster_size_and_shell or {2: 3, 3: 2, 4: 1}
    disordered_substructure = SQSTransformation._get_disordered_substructure(struct_disordered)
    clusters = {}
    for cluster_size, shell in cluster_size_and_shell.items():
        max_distance = SQSTransformation._get_max_neighbor_distance(disordered_substructure, shell)
        clusters[cluster_size] = max_distance + 0.01
    return clusters