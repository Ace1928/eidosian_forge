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
class XCFunctional(Section):
    """Defines the XC functional(s) to use."""

    def __init__(self, functionals: Iterable | None=None, keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        self.functionals = functionals or []
        keywords = keywords or {}
        subsections = subsections or {}
        location = 'CP2K_INPUT/FORCE_EVAL/DFT/XC/XC_FUNCTIONAL'
        for functional in self.functionals:
            subsections[functional] = Section(functional, subsections={}, repeats=False)
        super().__init__('XC_FUNCTIONAL', subsections=subsections, keywords=keywords, location=location, repeats=False, **kwargs)