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
class MVLRelax52Set(DictSet):
    """
    Implementation of VaspInputSet utilizing the public Materials Project
    parameters for INCAR & KPOINTS and VASP's recommended PAW potentials for
    POTCAR.

    Keynotes from VASP manual:
        1. Recommended potentials for calculations using vasp.5.2+
        2. If dimers with short bonds are present in the compound (O2, CO,
            N2, F2, P2, S2, Cl2), it is recommended to use the h potentials.
            Specifically, C_h, O_h, N_h, F_h, P_h, S_h, Cl_h
        3. Released on Oct 28, 2018 by VASP. Please refer to VASP
            Manual 1.2, 1.3 & 10.2.1 for more details.

    Args:
        structure (Structure): input structure.
        user_potcar_functional (str): choose from "PBE_52" and "PBE_54".
        **kwargs: Other kwargs supported by DictSet.
    """
    user_potcar_functional: UserPotcarFunctional = 'PBE_52'
    CONFIG = _load_yaml_config('MVLRelax52Set')
    _valid_potcars = ('PBE_52', 'PBE_54')