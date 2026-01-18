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
class OrbitalTransformation(Section):
    """
    Turns on the Orbital Transformation scheme for diagonalizing the Hamiltonian. Often faster
    and with guaranteed convergence compared to normal diagonalization, but requires the system
    to have a band gap.

    NOTE: OT has poor convergence for metallic systems and cannot use SCF mixing or smearing.
    Therefore, you should not use it for metals or systems with 'small' band gaps. In that
    case, use normal diagonalization
    """

    def __init__(self, minimizer: str='CG', preconditioner: str='FULL_ALL', algorithm: str='STRICT', rotation: bool=False, occupation_preconditioner: bool=False, energy_gap: float=-1, linesearch: str='2PNT', keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        """
        Initialize the OT section.

        Args:
            minimizer: The minimizer to use with the OT method. Default is conjugate gradient
                method, which is more robust, but more well-behaved systems should use DIIS, which
                can be as much as 50% faster.
            preconditioner: Preconditioner to use for OT, FULL_ALL tends to be most robust,
                but is not always most efficient. For difficult systems, FULL_SINGLE_INVERSE can be
                more robust, and is reasonably efficient with large systems. For huge, but well
                behaved, systems, where construction of the preconditioner can take a very long
                time, FULL_KINETIC can be a good choice.
            algorithm: What algorithm to use for OT. 'Strict': Taylor or diagonalization
                based algorithm. IRAC: Orbital Transformation based Iterative Refinement of the
                Approximate Congruence transformation (OT/IR).
            rotation: Introduce additional variables to allow subspace rotations (i.e fractional
                occupations)
            occupation_preconditioner: include the fractional occupation in the preconditioning
            energy_gap: Guess for the band gap. For FULL_ALL, should be smaller than the
                actual band gap, so simply using 0.01 is a robust value. Choosing a larger value
                will help if you start with a bad initial guess though. For FULL_SINGLE_INVERSE,
                energy_gap is treated as a lower bound. Values lower than 0.05 in this case can
                lead to stability issues.
            linesearch (str): From the manual: 1D line search algorithm to be used with the OT
                minimizer, in increasing order of robustness and cost. MINIMIZER CG combined with
                LINESEARCH GOLD should always find an electronic minimum. Whereas the 2PNT
                minimizer is almost always OK, 3PNT might be needed for systems in which successive
                OT CG steps do not decrease the total energy.
            keywords: additional keywords
            subsections: additional subsections
        """
        self.minimizer = minimizer
        self.preconditioner = preconditioner
        self.algorithm = algorithm
        self.rotation = rotation
        self.occupation_preconditioner = occupation_preconditioner
        self.energy_gap = energy_gap
        self.linesearch = linesearch
        keywords = keywords or {}
        subsections = subsections or {}
        description = "Sets the various options for the orbital transformation (OT) method. Default settings already provide an efficient, yet robust method. Most systems benefit from using the FULL_ALL preconditioner combined with a small value (0.001) of ENERGY_GAP. Well-behaved systems might benefit from using a DIIS minimizer. Advantages: It's fast, because no expensive diagonalizationis performed. If preconditioned correctly, method guaranteed to find minimum. Disadvantages: Sensitive to preconditioning. A good preconditioner can be expensive. No smearing, or advanced SCF mixing possible: POOR convergence for metallic systems."
        _keywords = {'MINIMIZER': Keyword('MINIMIZER', minimizer), 'PRECONDITIONER': Keyword('PRECONDITIONER', preconditioner), 'ENERGY_GAP': Keyword('ENERGY_GAP', energy_gap), 'ALGORITHM': Keyword('ALGORITHM', algorithm), 'LINESEARCH': Keyword('LINESEARCH', linesearch), 'ROTATION': Keyword('ROTATION', rotation), 'OCCUPATION_PRECONDITIONER': Keyword('OCCUPATION_PRECONDITIONER', occupation_preconditioner)}
        keywords.update(_keywords)
        super().__init__('OT', description=description, keywords=keywords, subsections=subsections, **kwargs)