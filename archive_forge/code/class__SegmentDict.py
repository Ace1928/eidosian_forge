from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.sets.base import AimsInputGenerator
from pymatgen.symmetry.bandstructure import HighSymmKpath
class _SegmentDict(TypedDict):
    coords: list[list[float]]
    labels: list[str]
    length: int