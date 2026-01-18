from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
@staticmethod
def _sites_to_mol(sites):
    """
        Return a ``Molecule`` object given a list of sites.

        Args:
            sites : A list of sites.

        Returns:
            mol (Molecule): A ``Molecule`` object.
        """
    return Molecule([site[0] for site in sites], [site[1] for site in sites])