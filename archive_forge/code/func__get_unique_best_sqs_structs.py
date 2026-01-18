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
def _get_unique_best_sqs_structs(sqs, best_only, return_ranked_list, remove_duplicate_structures, reduction_algo):
    """Gets unique sqs structures with lowest objective function. Requires an mcsqs output that has been run
            in parallel, otherwise returns Sqs.bestsqs.

        Args:
            sqs (Sqs): Sqs class object.
            best_only (bool): only return structures with lowest objective function.
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures.

                is returned. If False, only the single lowest energy structure is returned. Defaults to False.
            remove_duplicate_structures (bool): only return unique structures.
            reduction_algo (str): The lattice reduction algorithm to use. One of "niggli" or "LLL".
                reduction_algo=False does not reduce structure.

        Returns:
            list[dict[str, Structure | Callable]]: of the form {'structure': Structure, 'objective_function': ...},
                unless run in serial (returns a single structure Sqs.bestsqs)
        """
    if not return_ranked_list:
        return_struct = sqs.bestsqs
        if reduction_algo:
            return_struct = return_struct.get_reduced_structure(reduction_algo=reduction_algo)
        return return_struct
    structs = []
    for dct in sqs.allsqs:
        if not best_only or (best_only and dct['objective_function'] == sqs.objective_function):
            struct = dct['structure']
            struct.objective_function = dct['objective_function']
            structs.append(struct)
    if remove_duplicate_structures:
        matcher = StructureMatcher()
        unique_structs_grouped = matcher.group_structures(structs)
        structs = [group[0] for group in unique_structs_grouped]
    structs.sort(key=lambda x: x.objective_function if isinstance(x.objective_function, float) else -np.inf)
    to_return = [{'structure': struct, 'objective_function': struct.objective_function} for struct in structs]
    for dct in to_return:
        del dct['structure'].objective_function
        if reduction_algo:
            dct['structure'] = dct['structure'].get_reduced_structure(reduction_algo=reduction_algo)
    if isinstance(return_ranked_list, int) and (not isinstance(return_ranked_list, bool)):
        return to_return[:return_ranked_list]
    return to_return