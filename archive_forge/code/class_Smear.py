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
class Smear(Section):
    """Control electron smearing."""

    def __init__(self, elec_temp: float=300, method: str='FERMI_DIRAC', fixed_magnetic_moment: float=-100.0, keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        self.elec_temp = elec_temp
        self.method = method
        self.fixed_magnetic_moment = fixed_magnetic_moment
        keywords = keywords or {}
        subsections = subsections or {}
        description = 'Activates smearing of electron occupations'
        _keywords = {'ELEC_TEMP': Keyword('ELEC_TEMP', elec_temp), 'METHOD': Keyword('METHOD', method), 'FIXED_MAGNETIC_MOMENT': Keyword('FIXED_MAGNETIC_MOMENT', fixed_magnetic_moment)}
        keywords.update(_keywords)
        super().__init__('SMEAR', description=description, keywords=keywords, subsections=subsections, **kwargs)