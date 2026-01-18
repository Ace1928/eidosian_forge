import numpy as np
from ase import Atoms
from ase.cluster.util import get_element_info

    Return a decahedral cluster.

    Parameters
    ----------
    symbol: Chemical symbol (or atomic number) of the element.

    p: Number of atoms on the (100) facets perpendicular to the five
    fold axis.

    q: Number of atoms on the (100) facets parallel to the five fold
    axis. q = 1 corresponds to no visible (100) facets.

    r: Depth of the Marks re-entrence at the pentagon corners. r = 0
    corresponds to no re-entrence.

    latticeconstant (optional): The lattice constant. If not given,
    then it is extracted form ase.data.
    