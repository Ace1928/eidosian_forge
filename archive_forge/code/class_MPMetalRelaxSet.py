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
class MPMetalRelaxSet(DictSet):
    """
    Implementation of VaspInputSet utilizing parameters in the public
    Materials Project, but with tuning for metals. Key things are a denser
    k point density, and a.
    """
    CONFIG = MPRelaxSet.CONFIG

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        return {'ISMEAR': 1, 'SIGMA': 0.2}

    @property
    def kpoints_updates(self) -> dict | Kpoints:
        """Get updates to the kpoints configuration for this calculation type."""
        return {'reciprocal_density': 200}