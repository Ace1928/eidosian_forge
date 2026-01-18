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
class CifFile:
    """Reads and parses CifBlocks from a .cif file or string."""

    def __init__(self, data: dict, orig_string: str | None=None, comment: str | None=None) -> None:
        """
        Args:
            data (dict): Of CifBlock objects.
            orig_string (str): The original cif string.
            comment (str): Comment string.
        """
        self.data = data
        self.orig_string = orig_string
        self.comment = comment or '# generated using pymatgen'

    def __str__(self):
        out = '\n'.join(map(str, self.data.values()))
        return f'{self.comment}\n{out}\n'

    @classmethod
    def from_str(cls, string: str) -> Self:
        """Reads CifFile from a string.

        Args:
            string: String representation.

        Returns:
            CifFile
        """
        dct = {}
        for block_str in re.split('^\\s*data_', f'x\n{string}', flags=re.MULTILINE | re.DOTALL)[1:]:
            if 'powder_pattern' in re.split('\\n', block_str, maxsplit=1)[0]:
                continue
            block = CifBlock.from_str(f'data_{block_str}')
            dct[block.header] = block
        return cls(dct, string)

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """
        Reads CifFile from a filename.

        Args:
            filename: Filename

        Returns:
            CifFile
        """
        with zopen(str(filename), mode='rt', errors='replace') as file:
            return cls.from_str(file.read())