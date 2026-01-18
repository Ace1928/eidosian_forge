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
class MVLNPTMDSet(DictSet):
    """
    Class for writing a vasp md run in NPT ensemble.

    Notes:
        To eliminate Pulay stress, the default ENCUT is set to a rather large
        value of ENCUT, which is 1.5 * ENMAX.
    """
    start_temp: float = 0.0
    end_temp: float = 300.0
    nsteps: int = 1000
    time_step: float = 2
    spin_polarized: bool = False
    CONFIG = MITRelaxSet.CONFIG

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates = {'ALGO': 'Fast', 'ISIF': 3, 'LANGEVIN_GAMMA': [10] * self.structure.ntypesp, 'LANGEVIN_GAMMA_L': 1, 'MDALGO': 3, 'PMASS': 10, 'PSTRESS': 0, 'SMASS': 0, 'TEBEG': self.start_temp, 'TEEND': self.end_temp, 'NSW': self.nsteps, 'EDIFF_PER_ATOM': 1e-06, 'LSCALU': False, 'LCHARG': False, 'LPLANE': False, 'LWAVE': True, 'ISMEAR': 0, 'NELMIN': 4, 'LREAL': True, 'BMIX': 1, 'MAXMIX': 20, 'NELM': 500, 'NSIM': 4, 'ISYM': 0, 'IBRION': 0, 'NBLOCK': 1, 'KBLOCK': 100, 'POTIM': self.time_step, 'PREC': 'Low', 'ISPIN': 2 if self.spin_polarized else 1, 'LDAU': False}
        enmax = [self.potcar[i].keywords['ENMAX'] for i in range(self.structure.ntypesp)]
        updates['ENCUT'] = max(enmax) * 1.5
        return updates

    @property
    def kpoints_updates(self) -> Kpoints | dict:
        """Get updates to the kpoints configuration for this calculation type."""
        return Kpoints.gamma_automatic()