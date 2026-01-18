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
def _parse_sfac(self, line: str, it: Iterator[str]) -> ResSFAC:
    """Parses the SFAC block."""
    species = set(line.split())
    ions = []
    try:
        while True:
            line = next(it)
            if line == 'END':
                break
            ions.append(self._parse_ion(line))
    except StopIteration:
        raise ResParseError('Encountered end of file before END tag at end of SFAC block.')
    return ResSFAC(species, ions)