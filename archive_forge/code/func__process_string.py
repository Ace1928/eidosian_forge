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
@classmethod
def _process_string(cls, string):
    string = re.sub('(\\s|^)#.*$', '', string, flags=re.MULTILINE)
    string = re.sub('^\\s*\\n', '', string, flags=re.MULTILINE)
    string = string.encode('ascii', 'ignore').decode('ascii')
    deq = deque()
    multiline = False
    ml = []
    pattern = re.compile('([^\'"\\s][\\S]*)|\'(.*?)\'(?!\\S)|"(.*?)"(?!\\S)')
    for line in string.splitlines():
        if multiline:
            if line.startswith(';'):
                multiline = False
                deq.append(('', '', '', ' '.join(ml)))
                ml = []
                line = line[1:].strip()
            else:
                ml.append(line)
                continue
        if line.startswith(';'):
            multiline = True
            ml.append(line[1:].strip())
        else:
            for string in pattern.findall(line):
                deq.append(tuple(string))
    return deq