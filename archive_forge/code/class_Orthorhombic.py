from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
class Orthorhombic(BravaisLattice):
    """Abstract class for orthorhombic types."""

    def __init__(self, a, b, c):
        check_orc(a, b, c)
        BravaisLattice.__init__(self, a=a, b=b, c=c)