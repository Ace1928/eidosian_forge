from __future__ import annotations
import abc
import itertools
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast
from zipfile import ZipFile
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, SiteCollection, Species, Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
@dataclass
class MPNMRSet(DictSet):
    """Init a MPNMRSet.

    Args:
        structure (Structure): Structure from previous run.
        mode (str): The NMR calculation to run
            "cs": for Chemical Shift
            "efg" for Electric Field Gradient
        isotopes (list): list of Isotopes for quadrupole moments
        reciprocal_density (int): density of k-mesh by reciprocal volume.
        lepsilon (bool): Whether to add static dielectric calculation
        lcalcpol (bool): Whether to turn on evaluation of the Berry phase approximations
            for electronic polarization
        reciprocal_density (int): For static calculations, we usually set the
            reciprocal density by volume. This is a convenience arg to change
            that, rather than using user_kpoints_settings. Defaults to 100,
            which is ~50% more than that of standard relaxation calculations.
        small_gap_multiply ([float, float]): If the gap is less than
            1st index, multiply the default reciprocal_density by the 2nd
            index.
        **kwargs: kwargs supported by MPRelaxSet.
    """
    mode: Literal['cs', 'efg'] = 'cs'
    isotopes: list = field(default_factory=list)
    reciprocal_density: int = 100
    small_gap_multiply: tuple[float, float] | None = None
    inherit_incar: bool = True
    CONFIG = MPRelaxSet.CONFIG

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates: dict[str, Any] = {'NSW': 0, 'ISMEAR': -5, 'LCHARG': True, 'LORBIT': 11, 'LREAL': False}
        if self.mode.lower() == 'cs':
            updates.update(LCHIMAG=True, EDIFF=-1e-10, ISYM=0, LCHARG=False, LNMR_SYM_RED=True, NELMIN=10, NLSPLINE=True, PREC='ACCURATE', SIGMA=0.01)
        elif self.mode.lower() == 'efg':
            isotopes = {ist.split('-')[0]: ist for ist in self.isotopes}
            quad_efg = [float(Species(s.name).get_nmr_quadrupole_moment(isotopes.get(s.name))) for s in self.structure.species]
            updates.update(ALGO='FAST', EDIFF=-1e-10, ISYM=0, LCHARG=False, LEFG=True, QUAD_EFG=quad_efg, NELMIN=10, PREC='ACCURATE', SIGMA=0.01)
        return updates

    @property
    def kpoints_updates(self) -> dict:
        """Get updates to the kpoints configuration for this calculation type."""
        factor = 1.0
        if self.bandgap is not None and self.small_gap_multiply and (self.bandgap <= self.small_gap_multiply[0]):
            factor = self.small_gap_multiply[1]
        return {'reciprocal_density': self.reciprocal_density * factor}