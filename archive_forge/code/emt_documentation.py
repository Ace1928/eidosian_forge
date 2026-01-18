from math import sqrt, exp, log
import numpy as np
from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.neighborlist import NeighborList
from ase.calculators.calculator import (Calculator, all_changes,
Python implementation of the Effective Medium Potential.

    Supports the following standard EMT metals:
    Al, Cu, Ag, Au, Ni, Pd and Pt.

    In addition, the following elements are supported.
    They are NOT well described by EMT, and the parameters
    are not for any serious use:
    H, C, N, O

    The potential takes a single argument, ``asap_cutoff``
    (default: False).  If set to True, the cutoff mimics
    how Asap does it; most importantly the global cutoff
    is chosen from the largest atom present in the simulation,
    if False it is chosen from the largest atom in the parameter
    table.  True gives the behaviour of the Asap code and
    older EMT implementations, although the results are not
    bitwise identical.
    