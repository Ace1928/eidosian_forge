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
class BasisInfo(MSONable):
    """
    Summary info about a basis set.

    Attributes:
        electrons: Number of electrons
        core: Number of basis functions per core electron
        valence: Number of basis functions per valence electron OR number of exp if it
            is a FIT formatted admm basis
        polarization: Number of polarization functions
        diffuse: Number of added, diffuse/augmentation functions
        cc: Correlation consistent
        pc: Polarization consistent
        sr: Short-range optimized
        molopt: Optimized for molecules/solids
        admm: Whether this is an auxiliary basis set for ADMM
        lri: Whether this is a local resolution of identity auxiliary basis
        contracted: Whether this basis set is contracted
        xc: Exchange correlation functional used for creating this potential
    """
    electrons: int | None = None
    core: int | None = None
    valence: int | None = None
    polarization: int | None = None
    diffuse: int | None = None
    cc: bool | None = False
    pc: bool | None = False
    sr: bool | None = False
    molopt: bool | None = False
    admm: bool | None = False
    lri: bool | None = False
    contracted: bool | None = None
    xc: str | None = None

    def softmatch(self, other):
        """
        Soft matching to see if two basis sets match.

        Will only match those attributes which *are* defined for this basis info object (one way checking)
        """
        if not isinstance(other, BasisInfo):
            return False
        d1 = self.as_dict()
        d2 = other.as_dict()
        return all((not (v is not None and v != d2[k]) for k, v in d1.items()))

    @classmethod
    def from_str(cls, string: str) -> Self:
        """Get summary info from a string."""
        string = string.upper()
        data: dict[str, Any] = {}
        data['cc'] = 'CC' in string
        string = string.replace('CC', '')
        data['pc'] = 'PC' in string
        string = string.replace('PC', '')
        data['sr'] = 'SR' in string
        string = string.replace('SR', '')
        data['molopt'] = 'MOLOPT' in string
        string = string.replace('MOLOPT', '')
        for x in ('LDA', 'PADE', 'MGGA', 'GGA', 'HF', 'PBE0', 'PBE', 'BP', 'BLYP', 'B3LYP', 'SCAN'):
            if x in string:
                data['xc'] = x
                string = string.replace(x, '')
                break
        tmp = {'S': 1, 'D': 2, 'T': 3, 'Q': 4}
        if 'ADMM' in string or 'FIT' in string:
            data['admm'] = True
            bool_core = False
            data['contracted'] = 'C' in string
            nums = ''.join((s for s in string if s.isnumeric()))
            data['valence'] = int(nums) if nums else None
        else:
            data['admm'] = False
            if 'LRI' in string:
                data['lri'] = True
            bool_core = 'V' not in string or 'ALL' in string
        data['polarization'] = string.count('P')
        data['diffuse'] = string.count('X')
        string = f'#{string}'
        for idx, char in enumerate(string):
            if char == 'Z':
                z = int(tmp.get(string[idx - 1], string[idx - 1]))
                data['core'] = z if bool_core else None
                data['valence'] = z
            elif char == 'P' and string[idx - 1].isnumeric():
                data['polarization'] = int(string[idx - 1])
            elif char == 'X' and string[idx - 1].isnumeric():
                data['diffuse'] = int(string[idx - 1])
            elif char == 'Q' and string[idx + 1].isnumeric():
                data['electrons'] = int(''.join((_ for _ in string[idx + 1:] if _.isnumeric())))
        if not data['diffuse']:
            data['diffuse'] = string.count('AUG')
        return cls(**data)