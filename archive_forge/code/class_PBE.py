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
class PBE(Section):
    """Info about the PBE functional."""

    def __init__(self, parameterization: str='ORIG', scale_c: float=1, scale_x: float=1, keywords: dict | None=None, subsections: dict | None=None):
        """
        Args:
            parameterization (str):
                ORIG: original PBE
                PBESOL: PBE for solids/surfaces
                REVPBE: revised PBE
            scale_c (float): scales the correlation part of the functional.
            scale_x (float): scales the exchange part of the functional.
            keywords: additional keywords
            subsections: additional subsections.
        """
        self.parameterization = parameterization
        self.scale_c = scale_c
        self.scale_x = scale_x
        keywords = keywords or {}
        subsections = subsections or {}
        location = 'CP2K_INPUT/FORCE_EVAL/DFT/XC/XC_FUNCTIONAL/PBE'
        _keywords = {'PARAMETRIZATION': Keyword('PARAMETRIZATION', parameterization), 'SCALE_C': Keyword('SCALE_C', scale_c), 'SCALE_X': Keyword('SCALE_X', scale_x)}
        keywords.update(_keywords)
        super().__init__('PBE', subsections=subsections, repeats=False, location=location, section_parameters=[], keywords=keywords)