from typing import List
import numpy as np
from ase import Atoms
from .spacegroup import Spacegroup, _SPACEGROUP
def _has_spglib() -> bool:
    """Check if spglib is available"""
    try:
        import spglib
        assert spglib
    except ImportError:
        return False
    return True