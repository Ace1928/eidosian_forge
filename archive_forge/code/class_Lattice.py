import math
from typing import Optional, Sequence
import numpy as np
from ase.atoms import Atoms
import ase.data
class Lattice(Atoms, MillerInfo):
    """List of atoms initially containing a regular lattice of atoms.

    A part from the usual list of atoms methods this list of atoms type
    also has a method, `miller_to_direction`, used to convert from Miller
    indices to directions in the coordinate system of the lattice.
    """
    pass