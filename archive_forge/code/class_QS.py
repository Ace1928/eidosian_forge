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
class QS(Section):
    """Controls the quickstep settings (DFT driver)."""

    def __init__(self, method: str='GPW', eps_default: float=1e-10, eps_pgf_orb: float | None=None, extrapolation: str='ASPC', keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        """
        Initialize the QS Section.

        Args:
            method ("GPW" | "GAPW"): What DFT methodology to use. GPW (Gaussian Plane Waves) for
                DFT with pseudopotentials or GAPW (Gaussian Augmented Plane Waves) for all
                electron calculations.
            eps_default (float): The default level of convergence accuracy. NOTE: This is a
                global value for all the numerical value of all EPS_* values in QS module.
                It is not the same as EPS_SCF, which sets convergence accuracy of the SCF cycle
                alone.
            eps_pgf_orb: Precision for the overlap matrix. Default is to use sqrt(eps_default)
            extrapolation ("PS" | "ASPC"): Method use for extrapolation. If using
                gamma-point-only calculation, then one should either PS
                or ASPC (ASPC especially for MD runs). See the manual for other options.
            keywords: Additional keywords to add
            subsections: Subsections to initialize with.
        """
        self.method = method
        self.eps_default = eps_default
        self.eps_pgf_orb = eps_pgf_orb
        self.extrapolation = extrapolation
        keywords = keywords or {}
        subsections = subsections or {}
        description = 'Parameters needed to set up the Quickstep framework'
        _keywords = {'METHOD': Keyword('METHOD', self.method), 'EPS_DEFAULT': Keyword('EPS_DEFAULT', self.eps_default, description='Base precision level (in Ha)'), 'EXTRAPOLATION': Keyword('EXTRAPOLATION', self.extrapolation, description='WFN extrapolation between steps')}
        if eps_pgf_orb:
            _keywords['EPS_PGF_ORB'] = Keyword('EPS_PGF_ORB', self.eps_pgf_orb, description='Overlap matrix precision')
        keywords.update(_keywords)
        super().__init__('QS', description=description, keywords=keywords, subsections=subsections, **kwargs)