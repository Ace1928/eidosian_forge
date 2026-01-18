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
class LDOS(Section):
    """Controls printing of the LDOS (List-Density of states). i.e. projects onto specific atoms."""

    def __init__(self, index: int=1, alias: str | None=None, keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        """
        Initialize the LDOS section.

        Args:
            index: Index of the atom to project onto
            alias: section alias
            keywords: additional keywords
            subsections: additional subsections
        """
        self.index = index
        keywords = keywords or {}
        subsections = subsections or {}
        description = 'Controls printing of the projected density of states decomposed by atom type'
        _keywords = {'COMPONENTS': Keyword('COMPONENTS'), 'LIST': Keyword('LIST', index)}
        keywords.update(_keywords)
        super().__init__('LDOS', subsections=subsections, alias=alias, description=description, keywords=keywords, **kwargs)