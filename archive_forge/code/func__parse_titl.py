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
def _parse_titl(self, line: str) -> AirssTITL | None:
    """Parses the TITL entry. Checks for AIRSS values in the entry."""
    fields = line.split(maxsplit=6)
    if len(fields) >= 6:
        seed, pressure, volume, energy, spin, abs_spin = fields[:6]
        spg, nap = ('P1', '1')
        if len(fields) == 7:
            rest = fields[6]
            lp = rest.find('(')
            rp = rest.find(')')
            spg = rest[lp + 1:rp]
            nmin = rest.find('n -')
            nap = rest[nmin + 4:]
        return AirssTITL(seed, float(pressure), float(volume), float(energy), float(spin), float(abs_spin), spg, int(nap))
    return None