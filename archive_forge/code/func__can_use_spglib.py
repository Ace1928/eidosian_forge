from typing import List
import numpy as np
from ase import Atoms
from .spacegroup import Spacegroup, _SPACEGROUP
def _can_use_spglib(spacegroup: _SPACEGROUP=None) -> bool:
    """Helper dispatch function, for deciding if the spglib implementation
    can be used"""
    if not _has_spglib():
        return False
    if spacegroup is not None:
        return False
    return True