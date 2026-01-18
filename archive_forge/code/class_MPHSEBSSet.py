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
class MPHSEBSSet(DictSet):
    """
    Implementation of a VaspInputSet for HSE band structure computations.

    Remember that HSE band structures must be self-consistent in VASP. A band structure
    along symmetry lines for instance needs BOTH a uniform grid with appropriate weights
    AND a path along the lines with weight 0.

    Thus, the "uniform" mode is just like regular static SCF but allows adding custom
    kpoints (e.g., corresponding to known VBM/CBM) to the uniform grid that have zero
    weight (e.g., for better gap estimate).

    The "gap" mode behaves just like the "uniform" mode, however, if starting from a
    previous calculation, the VBM and CBM k-points will automatically be added to
    ``added_kpoints``.

    The "line" mode is just like "uniform" mode, but additionally adds k-points along
    symmetry lines with zero weight.

    The "uniform_dense" mode is like "uniform" mode but additionally adds a denser
    uniform mesh with zero weight. This can be useful when calculating Fermi surfaces
    or BoltzTraP/AMSET electronic transport using hybrid DFT.

    Args:
        structure (Structure): Structure to compute
        added_kpoints (list): a list of kpoints (list of 3 number list) added to the
            run. The k-points are in fractional coordinates
        mode (str): "Line" - generate k-points along symmetry lines for bandstructure.
            "Uniform" - generate uniform k-points grid.
        reciprocal_density (int): k-point density to use for uniform mesh.
        copy_chgcar (bool): Whether to copy the CHGCAR of a previous run.
        kpoints_line_density (int): k-point density for high symmetry lines
        dedos (float): Energy difference used to set NEDOS, based on the total energy
            range.
        optics (bool): Whether to add LOPTICS (used for calculating optical response).
        nbands_factor (float): Multiplicative factor for NBANDS when starting from a
            previous calculation. Choose a higher number if you are doing an LOPTICS
            calculation.
        **kwargs (dict): Any other parameters to pass into DictSet.
    """
    added_kpoints: list[Vector3D] = field(default_factory=list)
    mode: str = 'gap'
    reciprocal_density: float = 50
    copy_chgcar: bool = True
    kpoints_line_density: float = 20
    nbands_factor: float = 1.2
    zero_weighted_reciprocal_density: float = 100
    dedos: float = 0.02
    optics: bool = False
    CONFIG = MPHSERelaxSet.CONFIG

    def __post_init__(self) -> None:
        """Ensure mode is set correctly."""
        super().__post_init__()
        if 'reciprocal_density' in self.user_kpoints_settings:
            self.reciprocal_density = self.user_kpoints_settings['reciprocal_density']
        self.mode = self.mode.lower()
        supported_modes = ('line', 'uniform', 'gap', 'uniform_dense')
        if self.mode not in supported_modes:
            raise ValueError(f'Supported modes are: {', '.join(supported_modes)}')

    @property
    def kpoints_updates(self) -> dict:
        """Get updates to the kpoints configuration for this calculation type."""
        kpoints: dict[str, Any] = {'reciprocal_density': self.reciprocal_density, 'explicit': True}
        if self.mode == 'line':
            kpoints['zero_weighted_line_density'] = self.kpoints_line_density
        elif self.mode == 'uniform_dense':
            kpoints['zero_weighted_reciprocal_density'] = self.zero_weighted_reciprocal_density
        added_kpoints = deepcopy(self.added_kpoints)
        if self.prev_vasprun is not None and self.mode == 'gap':
            bs = self.prev_vasprun.get_band_structure()
            if not bs.is_metal():
                added_kpoints.append(bs.get_vbm()['kpoint'].frac_coords)
                added_kpoints.append(bs.get_cbm()['kpoint'].frac_coords)
        kpoints['added_kpoints'] = added_kpoints
        return kpoints

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates = dict(NSW=0, ISMEAR=0, SIGMA=0.05, ISYM=3, LCHARG=False, NELMIN=5)
        if self.mode == 'uniform' and len(self.added_kpoints) == 0:
            nedos = _get_nedos(self.prev_vasprun, self.dedos)
            updates.update({'ISMEAR': -5, 'NEDOS': nedos})
        else:
            updates.update({'ISMEAR': 0, 'SIGMA': 0.01})
        if self.prev_vasprun is not None:
            nbands = int(np.ceil(self.prev_vasprun.parameters['NBANDS'] * self.nbands_factor))
            updates['NBANDS'] = nbands
        if self.optics:
            updates.update({'LOPTICS': True, 'LREAL': False, 'CSHIFT': 1e-05})
        if self.prev_vasprun is not None and self.prev_outcar is not None:
            updates['ISPIN'] = _get_ispin(self.prev_vasprun, self.prev_outcar)
        return updates