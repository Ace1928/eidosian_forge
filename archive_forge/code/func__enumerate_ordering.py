from __future__ import annotations
import itertools
import logging
import math
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.transformation_abc import AbstractTransformation
def _enumerate_ordering(self, structure: Structure):
    struct = structure.copy()
    for indices, fraction in zip(self.indices, self.fractions):
        for ind in indices:
            new_sp = {sp: occu * fraction for sp, occu in structure[ind].species.items()}
            struct[ind] = new_sp
    from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation
    trans = EnumerateStructureTransformation()
    return trans.apply_transformation(struct, 10000)