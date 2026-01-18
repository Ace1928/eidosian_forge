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
def _get_ispin(vasprun: Vasprun | None, outcar: Outcar | None) -> int:
    """Get value of ISPIN depending on the magnetisation in the OUTCAR and vasprun."""
    if outcar is not None and outcar.magnetization is not None:
        site_magmom = np.array([i['tot'] for i in outcar.magnetization])
        return 2 if np.any(np.abs(site_magmom) > 0.02) else 1
    if vasprun is not None:
        return 2 if vasprun.is_spin else 1
    return 2