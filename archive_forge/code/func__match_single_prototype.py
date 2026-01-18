from __future__ import annotations
import os
from typing import TYPE_CHECKING
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.util.due import Doi, due
def _match_single_prototype(self, structure: Structure):
    sm = StructureMatcher(ltol=self.initial_ltol, stol=self.initial_stol, angle_tol=self.initial_angle_tol)
    tags = self._match_prototype(sm, structure)
    while len(tags) > 1:
        sm.ltol *= 0.8
        sm.stol *= 0.8
        sm.angle_tol *= 0.8
        tags = self._match_prototype(sm, structure)
        if sm.ltol < 0.01:
            break
    return tags