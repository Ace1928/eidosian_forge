from __future__ import annotations
import math
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from functools import partial
from inspect import getfullargspec
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.core import Composition, DummySpecies, Element, Lattice, PeriodicSite, Species, Structure, get_el_sp
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
def _loop_to_str(self, loop):
    out = 'loop_'
    for line in loop:
        out += '\n ' + line
    for fields in zip(*(self.data[k] for k in loop)):
        line = '\n'
        for val in map(self._format_field, fields):
            if val[0] == ';':
                out += line + '\n' + val
                line = '\n'
            elif len(line) + len(val) + 2 < self.max_len:
                line += '  ' + val
            else:
                out += line
                line = '\n  ' + val
        out += line
    return out