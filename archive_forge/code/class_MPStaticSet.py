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
class MPStaticSet(DictSet):
    """Creates input files for a static calculation.

    Args:
        structure (Structure): Structure from previous run.
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
    lepsilon: bool = False
    lcalcpol: bool = False
    reciprocal_density: int = 100
    small_gap_multiply: tuple[float, float] | None = None
    inherit_incar: bool = True
    CONFIG = MPRelaxSet.CONFIG

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates: dict[str, Any] = {'NSW': 0, 'ISMEAR': -5, 'LCHARG': True, 'LORBIT': 11, 'LREAL': False}
        if self.lepsilon:
            updates.update({'IBRION': 8, 'LEPSILON': True, 'LPEAD': True, 'NSW': 1, 'EDIFF': 1e-05})
        if self.lcalcpol:
            updates['LCALCPOL'] = True
        return updates

    @property
    def kpoints_updates(self) -> dict | Kpoints:
        """Get updates to the kpoints configuration for this calculation type."""
        factor = 1.0
        if self.bandgap is not None and self.small_gap_multiply and (self.bandgap <= self.small_gap_multiply[0]):
            factor = self.small_gap_multiply[1]
        if self.prev_kpoints and self.prev_kpoints.style == Kpoints.supported_modes.Monkhorst and (not self.lepsilon):
            kpoints = Kpoints.automatic_density_by_vol(self.structure, int(self.reciprocal_density * factor), self.force_gamma)
            k_div = [kp + 1 if kp % 2 == 1 else kp for kp in kpoints.kpts[0]]
            return Kpoints.monkhorst_automatic(k_div)
        return {'reciprocal_density': self.reciprocal_density * factor}