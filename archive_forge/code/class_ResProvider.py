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
class ResProvider(MSONable):
    """Provides access to elements of the res file in the form of familiar pymatgen objects."""

    def __init__(self, res: Res) -> None:
        """The :func:`from_str` and :func:`from_file` methods should be used instead of constructing this directly."""
        self._res = res

    @classmethod
    def _site_spin(cls, spin: float | None) -> dict[str, float] | None:
        """Check and return a dict with the site spin. Return None if spin is None."""
        if spin is None:
            return None
        return {'magmom': spin}

    @classmethod
    def from_str(cls, string: str) -> Self:
        """Construct a Provider from a string."""
        return cls(ResParser._parse_str(string))

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """Construct a Provider from a file."""
        return cls(ResParser._parse_file(filename))

    @property
    def rems(self) -> list[str]:
        """The full list of REM entries contained within the res file."""
        return self._res.REMS.copy()

    @property
    def lattice(self) -> Lattice:
        """Construct a Lattice from the res file."""
        cell = self._res.CELL
        return Lattice.from_parameters(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)

    @property
    def sites(self) -> list[PeriodicSite]:
        """Construct a list of PeriodicSites from the res file."""
        sfac_tag = self._res.SFAC
        return [PeriodicSite(ion.specie, ion.pos, self.lattice, properties=self._site_spin(ion.spin)) for ion in sfac_tag.ions]

    @property
    def structure(self) -> Structure:
        """Construct a Structure from the res file."""
        return Structure.from_sites(self.sites)