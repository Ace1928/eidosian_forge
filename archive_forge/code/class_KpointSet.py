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
class KpointSet(Section):
    """Specifies a kpoint line to be calculated between special points."""

    def __init__(self, npoints: int, kpoints: Iterable, units: str='B_VECTOR') -> None:
        """
        Args:
            npoints (int): Number of kpoints along the line.
            kpoints: A dictionary of {label: kpoint} kpoints defining the path
            units (str): Units for the kpoint coordinates.
                Options: "B_VECTOR" (reciprocal coordinates)
                         "CART_ANGSTROM" (units of 2*Pi/Angstrom)
                         "CART_BOHR" (units of 2*Pi/Bohr).
        """
        self.npoints = npoints
        self.kpoints = kpoints
        self.units = units
        keywords = {'NPOINTS': Keyword('NPOINTS', npoints), 'UNITS': Keyword('UNITS', units), 'SPECIAL_POINT': KeywordList([Keyword('SPECIAL_POINT', 'Gamma' if key.upper() == '\\GAMMA' else key, *kpt) for key, kpt in kpoints])}
        super().__init__(name='KPOINT_SET', subsections=None, repeats=True, description='Specifies a single k-point line for band structure calculations', keywords=keywords)