from typing import Dict, List, Tuple, Union, Optional
from numbers import Real
from collections import namedtuple
import re
from string import digits
import numpy as np
from ase import Atoms
from ase.units import Angstrom, Bohr, nm
def add_atom(self, name: str, pos: Tuple[float, float, float]) -> None:
    """Sets the symbol and position of an atom."""
    self.symbols.append(''.join([c for c in name if c not in digits]).capitalize())
    self.positions.append(pos)