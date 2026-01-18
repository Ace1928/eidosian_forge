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
def _format_field(self, val) -> str:
    val = str(val).strip()
    if len(val) > self.max_len:
        return f';\n{textwrap.fill(val, self.max_len)}\n;'
    if val == '':
        return '""'
    if (' ' in val or val[0] == '_') and (not (val[0] == "'" and val[-1] == "'")) and (not (val[0] == '"' and val[-1] == '"')):
        quote = '"' if "'" in val else "'"
        val = quote + val + quote
    return val