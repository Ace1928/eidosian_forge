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
class KeywordList(MSONable):
    """
    Some keywords can be repeated, which makes accessing them via the normal dictionary
    methods a little unnatural. This class deals with this by defining a collection
    of same-named keywords that are accessed by one name.
    """

    def __init__(self, keywords: Sequence[Keyword]):
        """
        Initializes a keyword list given a sequence of keywords.

        Args:
            keywords: A list of keywords. Must all have the same name (case-insensitive)
        """
        assert all((k.name.upper() == keywords[0].name.upper() for k in keywords)) if keywords else True
        self.name = keywords[0].name if keywords else None
        self.keywords = list(keywords)

    def __str__(self):
        return self.get_str()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return all((k == o for k, o in zip(self.keywords, other.keywords)))

    def __add__(self, other):
        return self.extend(other)

    def __len__(self):
        return len(self.keywords)

    def __getitem__(self, item):
        return self.keywords[item]

    def append(self, item):
        """Append the keyword list."""
        self.keywords.append(item)

    def extend(self, lst: Sequence[Keyword]) -> None:
        """Extend the keyword list."""
        self.keywords.extend(lst)

    def get_str(self, indent: int=0) -> str:
        """String representation of Keyword."""
        return ' \n'.join(('\t' * indent + str(k) for k in self.keywords))

    def verbosity(self, verbosity):
        """Silence all keywords in keyword list."""
        for k in self.keywords:
            k.verbosity(verbosity)