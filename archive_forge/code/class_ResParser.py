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
class ResParser:
    """Parser for the ShelX res file."""

    def __init__(self):
        self.line: int = 0
        self.filename: str | None = None
        self.source: str = ''

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

    def _parse_cell(self, line: str) -> ResCELL:
        """Parses the CELL entry."""
        fields = line.split()
        if len(fields) != 7:
            raise ResParseError(f'Failed to parse CELL line={line!r}, expected 7 fields.')
        field_1, a, b, c, alpha, beta, gamma = map(float, fields)
        return ResCELL(field_1, a, b, c, alpha, beta, gamma)

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

    @classmethod
    def _parse_str(cls, source: str) -> Res:
        """Parses the res file as a string."""
        self = cls()
        self.source = source
        return self._parse_txt()

    @classmethod
    def _parse_file(cls, filename: str | Path) -> Res:
        """Parses the res file as a file."""
        self = cls()
        with zopen(filename, mode='r') as file:
            self.source = file.read()
            return self._parse_txt()