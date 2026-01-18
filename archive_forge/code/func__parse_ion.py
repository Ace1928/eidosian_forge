from and back to a string/file is not guaranteed to be reversible, i.e. a diff on the output
from __future__ import annotations
import datetime
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element, Lattice, PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.core import ParseError
def _parse_ion(self, line: str) -> Ion:
    """Parses entries in the SFAC block."""
    fields = line.split()
    if len(fields) == 6:
        spin = None
    elif len(fields) == 7:
        spin = float(fields[-1])
    else:
        raise ResParseError(f'Failed to parse ion entry {line}, expected 6 or 7 fields.')
    specie = fields[0]
    specie_num = int(fields[1])
    x, y, z, occ = map(float, fields[2:6])
    return Ion(specie, specie_num, (x, y, z), occ, spin)