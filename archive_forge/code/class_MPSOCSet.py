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
class MPSOCSet(DictSet):
    """An input set for running spin-orbit coupling (SOC) calculations.

    Args:
        structure (Structure): the structure must have the 'magmom' site
            property and each magnetic moment value must have 3
            components. eg: ``magmom = [[0,0,2], ...]``
        saxis (tuple): magnetic moment orientation
        copy_chgcar: Whether to copy the old CHGCAR. Defaults to True.
        nbands_factor (float): Multiplicative factor for NBANDS. Choose a
            higher number if you are doing an LOPTICS calculation.
        reciprocal_density (int): density of k-mesh by reciprocal volume.
        small_gap_multiply ([float, float]): If the gap is less than
            1st index, multiply the default reciprocal_density by the 2nd
            index.
        lepsilon (bool): Whether to add static dielectric calculation
        lcalcpol (bool): Whether to turn on evaluation of the Berry phase approximations
            for electronic polarization
        magmom (list[list[float]]): Override for the structure magmoms.
        **kwargs: kwargs supported by DictSet.
    """
    saxis: tuple[int, int, int] = (0, 0, 1)
    nbands_factor: float = 1.2
    lepsilon: bool = False
    lcalcpol: bool = False
    reciprocal_density: float = 100
    small_gap_multiply: tuple[float, float] | None = None
    magmom: list[Vector3D] | None = None
    inherit_incar: bool = True
    copy_chgcar: bool = True
    CONFIG = MPRelaxSet.CONFIG

    def __post_init__(self):
        super().__post_init__()
        if self.structure and (not hasattr(self.structure[0], 'magmom')) and (not isinstance(self.structure[0].magmom, list)):
            raise ValueError("The structure must have the 'magmom' site property and each magnetic moment value must have 3 components. e.g. magmom = [0,0,2]")

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates = {'ISYM': -1, 'LSORBIT': 'T', 'ICHARG': 11, 'SAXIS': list(self.saxis), 'NSW': 0, 'ISMEAR': -5, 'LCHARG': True, 'LORBIT': 11, 'LREAL': False}
        if self.lepsilon:
            updates.update({'IBRION': 8, 'LEPSILON': True, 'LPEAD': True, 'NSW': 1})
        if self.lcalcpol:
            updates['LCALCPOL'] = True
        if self.prev_vasprun is not None:
            n_bands = int(np.ceil(self.prev_vasprun.parameters['NBANDS'] * self.nbands_factor))
            updates['NBANDS'] = n_bands
        return updates

    @property
    def kpoints_updates(self) -> dict:
        """Get updates to the kpoints configuration for this calculation type."""
        factor = 1.0
        if self.bandgap is not None and self.small_gap_multiply and (self.bandgap <= self.small_gap_multiply[0]):
            factor = self.small_gap_multiply[1]
        return {'reciprocal_density': self.reciprocal_density * factor}

    @DictSet.structure.setter
    def structure(self, structure: Structure | None) -> None:
        if structure is not None:
            if self.magmom:
                structure = structure.copy(site_properties={'magmom': self.magmom})
            if hasattr(structure[0], 'magmom'):
                if not isinstance(structure[0].magmom, list):
                    structure = structure.copy(site_properties={'magmom': [[0, 0, site.magmom] for site in structure]})
            else:
                raise ValueError('Neither the previous structure has magmom property nor magmom provided')
        DictSet.structure.fset(self, structure)