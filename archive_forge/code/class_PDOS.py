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
class PDOS(Section):
    """
    Controls printing of projected density of states onto the different atom KINDS
    (elemental decomposed DOS).
    """

    def __init__(self, nlumo: int=-1, keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        """
        Initialize the PDOS section.

        Args:
            nlumo: how many unoccupied orbitals to include (-1==ALL)
            keywords: additional keywords
            subsections: additional subsections
        """
        self.nlumo = nlumo
        keywords = keywords or {}
        subsections = subsections or {}
        description = 'Controls printing of the projected density of states'
        _keywords = {'NLUMO': Keyword('NLUMO', nlumo), 'COMPONENTS': Keyword('COMPONENTS')}
        keywords.update(_keywords)
        super().__init__('PDOS', description=description, keywords=keywords, subsections=subsections, **kwargs)