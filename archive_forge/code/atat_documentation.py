from __future__ import annotations
import numpy as np
from pymatgen.core import Lattice, Structure, get_el_sp

        Parses a rndstr.in, lat.in or bestsqs.out file into pymatgen's
        Structure format.

        Args:
            data: contents of a rndstr.in, lat.in or bestsqs.out file

        Returns:
            Structure object
        