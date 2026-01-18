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
class DftPlusU(Section):
    """Controls DFT+U for an atom kind."""

    def __init__(self, eps_u_ramping=1e-05, init_u_ramping_each_scf=False, l=-1, u_minus_j=0, u_ramping=0, keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        """
        Initialize the DftPlusU section.

        Args:
            eps_u_ramping: (float) SCF convergence threshold at which to start ramping the U value
            init_u_ramping_each_scf: (bool) Whether or not to do u_ramping each scf cycle
            l: (int) angular moment of the orbital to apply the +U correction
            u_minus_j: (float) the effective U parameter, Ueff = U-J
            u_ramping: (float) stepwise amount to increase during ramping until u_minus_j is reached
            keywords: additional keywords
            subsections: additional subsections
        """
        name = 'DFT_PLUS_U'
        self.eps_u_ramping = 1e-05
        self.init_u_ramping_each_scf = False
        self.l = l
        self.u_minus_j = u_minus_j
        self.u_ramping = u_ramping
        keywords = keywords or {}
        subsections = subsections or {}
        description = 'Settings for on-site Hubbard +U correction for this atom kind.'
        _keywords = {'EPS_U_RAMPING': Keyword('EPS_U_RAMPING', eps_u_ramping), 'INIT_U_RAMPING_EACH_SCF': Keyword('INIT_U_RAMPING_EACH_SCF', init_u_ramping_each_scf), 'L': Keyword('L', l), 'U_MINUS_J': Keyword('U_MINUS_J', u_minus_j), 'U_RAMPING': Keyword('U_RAMPING', u_ramping)}
        keywords.update(_keywords)
        super().__init__(name=name, subsections=None, description=description, keywords=keywords, **kwargs)