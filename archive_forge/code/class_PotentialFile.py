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
@dataclass
class PotentialFile(DataFile):
    """Data file for potentials only."""

    @classmethod
    def from_str(cls, string: str) -> Self:
        """Initialize from a string representation."""
        basis_sets = [GthPotential.from_str(c) for c in chunk(string)]
        return cls(objects=basis_sets)