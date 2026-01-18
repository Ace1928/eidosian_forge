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
@classmethod
def _sfac_from_sites(cls, sites: list[PeriodicSite]) -> ResSFAC:
    """Produce a SFAC block from a list of pymatgen PeriodicSite."""
    ions = cls._ions_from_sites(sites)
    species = {ion.specie for ion in ions}
    return ResSFAC(species, ions)