from pathlib import Path
from re import compile
import numpy as np
from ase import Atoms
from ase.utils import reader
from ase.units import Bohr
def _is_block(val):
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
        return True
    return False