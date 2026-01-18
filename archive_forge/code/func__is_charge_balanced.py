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
def _is_charge_balanced(struct) -> bool:
    """Checks if the structure object is charge balanced."""
    return abs(sum((site.specie.oxi_state for site in struct))) < Substitutor.charge_balanced_tol