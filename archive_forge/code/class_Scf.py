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
class Scf(Section):
    """Controls the self consistent field loop."""

    def __init__(self, max_scf: int=50, eps_scf: float=1e-06, scf_guess: str='RESTART', keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        """
        Initialize the Scf section.

        Args:
            max_scf (int): Maximum number of SCF loops before terminating. Defaults to 50.
            eps_scf (float): Convergence criteria for SCF loop. Defaults to 1e-6.
            scf_guess: Initial guess for SCF loop.
                "ATOMIC": Generate an atomic density using the atomic code
                "CORE": Diagonalize the core Hamiltonian for an initial guess.
                "HISTORY_RESTART": Extrapolated from previous RESTART files.
                "MOPAC": Use same guess as MOPAC for semi-empirical methods or a simple
                    diagonal density matrix for other methods.
                "NONE": Skip initial guess (only for NON-SCC DFTB).
                "RANDOM": Use random wavefunction coefficients.
                "RESTART": Use the RESTART file as an initial guess (and ATOMIC if not present).
                "SPARSE": Generate a sparse wavefunction using the atomic code (for OT based
                    methods).
            keywords: Additional keywords
            subsections: Additional subsections
        """
        self.max_scf = max_scf
        self.eps_scf = eps_scf
        self.scf_guess = scf_guess
        keywords = keywords or {}
        subsections = subsections or {}
        description = 'Parameters needed to perform an SCF run.'
        _keywords = {'MAX_SCF': Keyword('MAX_SCF', max_scf, description='Max number of steps for an inner SCF loop'), 'EPS_SCF': Keyword('EPS_SCF', eps_scf, description='Convergence threshold for SCF'), 'SCF_GUESS': Keyword('SCF_GUESS', scf_guess, description='How to initialize the density matrix'), 'MAX_ITER_LUMO': Keyword('MAX_ITER_LUMO', kwargs.get('max_iter_lumo', 400), description='Iterations for solving for unoccupied levels when running OT')}
        keywords.update(_keywords)
        super().__init__('SCF', description=description, keywords=keywords, subsections=subsections, **kwargs)