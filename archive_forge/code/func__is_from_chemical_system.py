from __future__ import annotations
import functools
import itertools
import logging
from operator import mul
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.alchemy.filters import RemoveDuplicatesFilter, RemoveExistingFilter
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.alchemy.transmuters import StandardTransmuter
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability
from pymatgen.core import get_el_sp
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.util.due import Doi, due
@staticmethod
def _is_from_chemical_system(chemical_system, struct):
    """Checks if the structure object is from the given chemical system."""
    return {sp.symbol for sp in struct.composition} == set(chemical_system)