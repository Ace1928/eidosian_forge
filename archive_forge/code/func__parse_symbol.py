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
def _parse_symbol(self, sym):
    """
        Parse a string with a symbol to extract a string representing an element.

        Args:
            sym (str): A symbol to be parsed.

        Returns:
            A string with the parsed symbol. None if no parsing was possible.
        """
    special = {'Hw': 'H', 'Ow': 'O', 'Wat': 'O', 'wat': 'O', 'OH': '', 'OH2': '', 'NO3': 'N'}
    parsed_sym = None
    m_sp = re.match('|'.join(special), sym)
    if m_sp:
        parsed_sym = special[m_sp.group()]
    elif Element.is_valid_symbol(sym[:2].title()):
        parsed_sym = sym[:2].title()
    elif Element.is_valid_symbol(sym[0].upper()):
        parsed_sym = sym[0].upper()
    else:
        m = re.match('w?[A-Z][a-z]*', sym)
        if m:
            parsed_sym = m.group()
    if parsed_sym is not None and (m_sp or not re.match(f'{parsed_sym}\\d*', sym)):
        msg = f'{sym} parsed as {parsed_sym}'
        warnings.warn(msg)
        self.warnings.append(msg)
    return parsed_sym