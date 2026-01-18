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
class PotentialInfo(MSONable):
    """
    Metadata for this potential.

    Attributes:
        electrons: Total number of electrons
        potential_type: Potential type (e.g. GTH)
        nlcc: Nonlinear core corrected potential
        xc: Exchange correlation functional used for creating this potential
    """
    electrons: int | None = None
    potential_type: str | None = None
    nlcc: bool | None = None
    xc: str | None = None

    def softmatch(self, other):
        """
        Soft matching to see if two potentials match.

        Will only match those attributes which *are* defined for this basis info object (one way checking)
        """
        if not isinstance(other, PotentialInfo):
            return False
        d1 = self.as_dict()
        d2 = other.as_dict()
        return all((not (v is not None and v != d2[k]) for k, v in d1.items()))

    @classmethod
    def from_str(cls, string: str) -> Self:
        """Get a cp2k formatted string representation."""
        string = string.upper()
        data: dict[str, Any] = {}
        if 'NLCC' in string:
            data['nlcc'] = True
        if 'GTH' in string:
            data['potential_type'] = 'GTH'
        for idx, char in enumerate(string, start=1):
            if char == 'Q' and string[idx].isnumeric():
                data['electrons'] = int(''.join((_ for _ in string[idx:] if _.isnumeric())))
        for x in ('LDA', 'PADA', 'MGGA', 'GGA', 'HF', 'PBE0', 'PBE', 'BP', 'BLYP', 'B3LYP', 'SCAN'):
            if x in string:
                data['xc'] = x
                break
        return cls(**data)