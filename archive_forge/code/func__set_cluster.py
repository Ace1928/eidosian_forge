from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from tabulate import tabulate
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.core import ParseError
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
def _set_cluster(self):
    """
        Compute and set the cluster of atoms as a Molecule object. The site
        coordinates are translated such that the absorbing atom (aka central
        atom) is at the origin.

        Returns:
            Molecule
        """
    center = self.struct[self.center_index].coords
    sphere = self.struct.get_neighbors(self.struct[self.center_index], self.radius)
    symbols = [self.absorbing_atom]
    coords = [[0, 0, 0]]
    for site_dist in sphere:
        site_symbol = re.sub('[^aA-zZ]+', '', site_dist[0].species_string)
        symbols.append(site_symbol)
        coords.append(site_dist[0].coords - center)
    return Molecule(symbols, coords)