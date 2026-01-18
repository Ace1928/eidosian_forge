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
class Mgrid(Section):
    """Controls the multigrid for numerical integration."""

    def __init__(self, cutoff: float=1200, rel_cutoff: float=80, ngrids: int=5, progression_factor: int=3, keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        """
        Initialize the MGRID section.

        Args:
            cutoff: Cutoff energy (in Rydbergs for historical reasons) defining how find of
                Gaussians will be used
            rel_cutoff: The relative cutoff energy, which defines how to map the Gaussians onto
                the multigrid. If the value is too low then, even if you have a high cutoff
                with sharp Gaussians, they will be mapped to the course part of the multigrid
            ngrids: number of grids to use
            progression_factor: divisor that decides how to map Gaussians the multigrid after
                the highest mapping is decided by rel_cutoff
            keywords: additional keywords
            subsections: additional subsections
        """
        self.cutoff = cutoff
        self.rel_cutoff = rel_cutoff
        self.ngrids = ngrids
        self.progression_factor = progression_factor
        keywords = keywords or {}
        subsections = subsections or {}
        description = 'Multigrid information. Multigrid allows for sharp gaussians and diffuse gaussians to be treated on different grids, where the spacing of FFT integration points can be tailored to the degree of sharpness/diffusiveness'
        _keywords = {'CUTOFF': Keyword('CUTOFF', cutoff, description='Cutoff in [Ry] for finest level of the MG.'), 'REL_CUTOFF': Keyword('REL_CUTOFF', rel_cutoff, description='Controls which gaussians are mapped to which level of the MG'), 'NGRIDS': Keyword('NGRIDS', ngrids, description='Number of grid levels in the MG'), 'PROGRESSION_FACTOR': Keyword('PROGRESSION_FACTOR', progression_factor)}
        keywords.update(_keywords)
        super().__init__('MGRID', description=description, keywords=keywords, subsections=subsections, **kwargs)