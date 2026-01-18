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
def _parse_txt(self) -> Res:
    """Parses the text of the file."""
    _REMS: list[str] = []
    _TITL: AirssTITL | None = None
    _CELL: ResCELL | None = None
    _SFAC: ResSFAC | None = None
    txt = self.source
    it = iter(txt.splitlines())
    try:
        while True:
            line = next(it)
            self.line += 1
            split = line.split(maxsplit=1)
            splits = len(split)
            if splits == 0:
                continue
            if splits == 1:
                first, rest = (*split, '')
            else:
                first, rest = split
            if first == 'TITL':
                _TITL = self._parse_titl(rest)
            elif first == 'REM':
                _REMS.append(rest)
            elif first == 'CELL':
                _CELL = self._parse_cell(rest)
            elif first == 'LATT':
                pass
            elif first == 'SFAC':
                _SFAC = self._parse_sfac(rest, it)
            else:
                raise Warning(f'Skipping line={line!r}, tag {first} not recognized.')
    except StopIteration:
        pass
    if _CELL is None or _SFAC is None:
        raise ResParseError('Did not encounter CELL or SFAC entry when parsing.')
    return Res(_TITL, _REMS, _CELL, _SFAC)