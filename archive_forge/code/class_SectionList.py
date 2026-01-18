from __future__ import annotations
import abc
import copy
import hashlib
import itertools
import os
import re
import textwrap
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.io.cp2k.utils import chunk, postprocessor, preprocessor
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class SectionList(MSONable):
    """Section list."""

    def __init__(self, sections: Sequence[Section]):
        """
        Initializes a SectionList object using a sequence of sections.

        Args:
            sections: A list of keywords. Must all have the same name (case-insensitive)
        """
        assert all((k.name.upper() == sections[0].name.upper() for k in sections)) if sections else True
        self.name = sections[0].name if sections else None
        self.alias = sections[0].alias if sections else None
        self.sections = list(sections)

    def __str__(self):
        return self.get_str()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SectionList):
            return NotImplemented
        return all((k == o for k, o in zip(self.sections, other.sections)))

    def __add__(self, other):
        self.append(other)
        return self

    def __len__(self):
        return len(self.sections)

    def __getitem__(self, item):
        return self.sections[item]

    def __deepcopy__(self, memodict=None):
        return SectionList(sections=[d.__deepcopy__() for d in self.sections])

    @staticmethod
    def _get_str(d, indent=0):
        return ' \n'.join((s._get_str(s, indent) for s in d))

    def get_str(self) -> str:
        """Return string representation of section list."""
        return SectionList._get_str(self.sections)

    def get(self, d, index=-1):
        """
        Get for section list. If index is specified, return the section at that index.
        Otherwise, return a get on the last section.
        """
        return self.sections[index].get(d)

    def append(self, item) -> None:
        """Append the section list."""
        self.sections.append(item)

    def extend(self, lst: list) -> None:
        """Extend the section list."""
        self.sections.extend(lst)

    def verbosity(self, verbosity) -> None:
        """Silence all sections in section list."""
        for k in self.sections:
            k.verbosity(verbosity)