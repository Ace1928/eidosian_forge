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
def _set_previous(self, prev_dir: str | Path | None=None):
    """Load previous calculation outputs."""
    if prev_dir is not None:
        vasprun, outcar = get_vasprun_outcar(prev_dir)
        self.prev_vasprun = vasprun
        self.prev_outcar = outcar
        self.prev_incar = vasprun.incar
        self.prev_kpoints = Kpoints.from_dict(vasprun.kpoints.as_dict())
        if vasprun.efermi is None:
            vasprun.efermi = outcar.efermi
        bs = vasprun.get_band_structure(efermi='smart')
        self.bandgap = 0 if bs.is_metal() else bs.get_band_gap()['energy']
        self.structure = get_structure_from_prev_run(vasprun, outcar)