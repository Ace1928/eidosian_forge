from __future__ import annotations
import os
from typing import TYPE_CHECKING
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.util.due import Doi, due
@staticmethod
def _match_prototype(structure_matcher, structure):
    tags = []
    for d in AFLOW_PROTOTYPE_LIBRARY:
        struct = d['snl'].structure
        match = structure_matcher.fit_anonymous(struct, structure)
        if match:
            tags.append(d)
    return tags