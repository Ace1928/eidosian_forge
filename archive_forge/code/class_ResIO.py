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
class ResIO:
    """
    Class providing convenience methods for converting a Structure or ComputedStructureEntry
    to/from a string or file in the res format as used by AIRSS.

    Note: Converting from and back to pymatgen objects is expected to be reversible, i.e. you
    should get the same Structure or ComputedStructureEntry back. On the other hand, converting
    from and back to a string/file is not guaranteed to be reversible, i.e. a diff on the output
    would not be empty. The difference should be limited to whitespace, float precision, and the
    REM entries.

    If the TITL entry doesn't exist or is malformed or empty, then you can only get
    a Structure. Attempting to get an Entry will raise a ResError.
    """

    @classmethod
    def structure_from_str(cls, string: str) -> Structure:
        """Produces a pymatgen Structure from contents of a res file."""
        return ResProvider.from_str(string).structure

    @classmethod
    def structure_from_file(cls, filename: str) -> Structure:
        """Produces a pymatgen Structure from a res file."""
        return ResProvider.from_file(filename).structure

    @classmethod
    def structure_to_str(cls, structure: Structure) -> str:
        """Produce the contents of a res file from a pymatgen Structure."""
        return str(ResWriter(structure))

    @classmethod
    def structure_to_file(cls, structure: Structure, filename: str) -> None:
        """Write a pymatgen Structure to a res file."""
        return ResWriter(structure).write(filename)

    @classmethod
    def entry_from_str(cls, string: str) -> ComputedStructureEntry:
        """Produce a pymatgen ComputedStructureEntry from contents of a res file."""
        return AirssProvider.from_str(string).entry

    @classmethod
    def entry_from_file(cls, filename: str) -> ComputedStructureEntry:
        """Produce a pymatgen ComputedStructureEntry from a res file."""
        return AirssProvider.from_file(filename).entry

    @classmethod
    def entry_to_str(cls, entry: ComputedStructureEntry) -> str:
        """Produce the contents of a res file from a pymatgen ComputedStructureEntry."""
        return str(ResWriter(entry))

    @classmethod
    def entry_to_file(cls, entry: ComputedStructureEntry, filename: str) -> None:
        """Write a pymatgen ComputedStructureEntry to a res file."""
        return ResWriter(entry).write(filename)