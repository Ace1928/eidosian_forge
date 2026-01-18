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
def _site_spin(cls, spin: float | None) -> dict[str, float] | None:
    """Check and return a dict with the site spin. Return None if spin is None."""
    if spin is None:
        return None
    return {'magmom': spin}