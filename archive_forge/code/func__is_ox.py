from __future__ import annotations
import os
import warnings
import numpy as np
from monty.serialization import loadfn
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
def _is_ox(structure):
    for elem in structure.composition:
        try:
            elem.oxi_state
        except AttributeError:
            return False
    return True