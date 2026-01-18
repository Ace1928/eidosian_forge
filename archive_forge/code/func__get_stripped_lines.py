from pathlib import Path
from re import compile
import numpy as np
from ase import Atoms
from ase.utils import reader
from ase.units import Bohr
def _get_stripped_lines(fd):
    return [_f for _f in [L.split('#')[0].strip() for L in fd] if _f]