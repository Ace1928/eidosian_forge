from __future__ import annotations
import re
from typing import TYPE_CHECKING
from monty.io import zopen
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

        Reads a CSSR file to a Cssr object.

        Args:
            filename (str): Filename to read from.

        Returns:
            Cssr object.
        