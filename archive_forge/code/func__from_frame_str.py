from __future__ import annotations
import re
from io import StringIO
from typing import TYPE_CHECKING, cast
import pandas as pd
from monty.io import zopen
from pymatgen.core import Molecule, Structure
from pymatgen.core.structure import SiteCollection
@staticmethod
def _from_frame_str(contents) -> Molecule:
    """Convert a single frame XYZ string to a molecule."""
    lines = contents.split('\n')
    num_sites = int(lines[0])
    coords = []
    sp = []
    coord_pattern = re.compile('(\\w+)\\s+([0-9\\-\\+\\.*^eEdD]+)\\s+([0-9\\-\\+\\.*^eEdD]+)\\s+([0-9\\-\\+\\.*^eEdD]+)')
    for i in range(2, 2 + num_sites):
        m = coord_pattern.search(lines[i])
        if m:
            sp.append(m.group(1))
            xyz = [val.lower().replace('d', 'e').replace('*^', 'e') for val in m.groups()[1:4]]
            coords.append([float(val) for val in xyz])
    return Molecule(sp, coords)