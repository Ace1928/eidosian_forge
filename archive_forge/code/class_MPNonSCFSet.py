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
class MPNonSCFSet(DictSet):
    """
    Init a MPNonSCFSet. Typically, you would use the classmethod
    from_prev_calc to initialize from a previous SCF run.

    Args:
        structure (Structure): Structure to compute
        mode (str): Line, Uniform or Boltztrap mode supported.
        nedos (int): nedos parameter. Default to 2001.
        dedos (float): setting nedos=0 and uniform mode in from_prev_calc,
            an automatic nedos will be calculated using the total energy range
            divided by the energy step dedos
        reciprocal_density (int): density of k-mesh by reciprocal
            volume (defaults to 100)
        kpoints_line_density (int): Line density for Line mode.
        optics (bool): whether to add dielectric function
        copy_chgcar: Whether to copy the old CHGCAR when starting from a
            previous calculation.
        nbands_factor (float): Multiplicative factor for NBANDS when starting
            from a previous calculation. Choose a higher number if you are
            doing an LOPTICS calculation.
        small_gap_multiply ([float, float]): When starting from a previous
            calculation, if the gap is less than 1st index, multiply the default
            reciprocal_density by the 2nd index.
        **kwargs: kwargs supported by MPRelaxSet.
    """
    mode: str = 'line'
    nedos: int = 2001
    dedos: float = 0.005
    reciprocal_density: float = 100
    kpoints_line_density: float = 20
    optics: bool = False
    copy_chgcar: bool = True
    nbands_factor: float = 1.2
    small_gap_multiply: tuple[float, float] | None = None
    inherit_incar: bool = True
    CONFIG = MPRelaxSet.CONFIG

    def __post_init__(self):
        """Perform inputset validation."""
        super().__post_init__()
        mode = self.mode = self.mode.lower()
        valid_modes = ('line', 'uniform', 'boltztrap')
        if mode not in valid_modes:
            raise ValueError(f'Invalid mode={mode!r}. Supported modes for NonSCF runs are {', '.join(map(repr, valid_modes))}')
        if (mode != 'uniform' or self.nedos < 2000) and self.optics:
            warnings.warn('It is recommended to use Uniform mode with a high NEDOS for optics calculations.')
        if self.standardize:
            warnings.warn('Use of standardize=True with from_prev_run is not recommended as there is no guarantee the copied files will be appropriate for the standardized structure. copy_chgcar is enforced to be false.')
            self.copy_chgcar = False

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates: dict[str, Any] = {'LCHARG': False, 'LORBIT': 11, 'LWAVE': False, 'NSW': 0, 'ISYM': 0, 'ICHARG': 11}
        if self.prev_vasprun is not None:
            n_bands = int(np.ceil(self.prev_vasprun.parameters['NBANDS'] * self.nbands_factor))
            updates['NBANDS'] = n_bands
        nedos = _get_nedos(self.prev_vasprun, self.dedos) if self.nedos == 0 else self.nedos
        if self.mode == 'uniform':
            updates.update({'ISMEAR': -5, 'ISYM': 2, 'NEDOS': nedos})
        elif self.mode in ('line', 'boltztrap'):
            sigma = 0.2 if self.bandgap == 0 or self.bandgap is None else 0.01
            updates.update({'ISMEAR': 0, 'SIGMA': sigma})
        if self.optics:
            updates.update({'LOPTICS': True, 'LREAL': False, 'CSHIFT': 1e-05, 'NEDOS': nedos})
        if self.prev_vasprun is not None and self.prev_outcar is not None:
            updates['ISPIN'] = _get_ispin(self.prev_vasprun, self.prev_outcar)
        updates['MAGMOM'] = None
        return updates

    @property
    def kpoints_updates(self) -> dict:
        """Get updates to the kpoints configuration for this calculation type."""
        factor = 1.0
        if self.bandgap is not None and self.small_gap_multiply and (self.bandgap <= self.small_gap_multiply[0]):
            factor = self.small_gap_multiply[1]
        if self.mode == 'line':
            return {'line_density': self.kpoints_line_density * factor}
        if self.mode == 'boltztrap':
            return {'explicit': True, 'reciprocal_density': self.reciprocal_density * factor}
        return {'reciprocal_density': self.reciprocal_density * factor}