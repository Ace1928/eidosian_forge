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
def _remove_unused_incar_params(incar, skip: Sequence[str]=()) -> None:
    """
    Remove INCAR parameters that are not actively used by VASP.

    Args:
        incar (dict): An incar.
        skip (list of str): Keys to skip.
    """
    opt_flags = ['EDIFFG', 'IBRION', 'ISIF', 'POTIM']
    if incar.get('NSW', 0) == 0:
        for opt_flag in opt_flags:
            if opt_flag not in skip:
                incar.pop(opt_flag, None)
    if incar.get('ISPIN', 1) == 1 and 'MAGMOM' not in skip:
        incar.pop('MAGMOM', None)
    ldau_flags = ['LDAUU', 'LDAUJ', 'LDAUL', 'LDAUTYPE']
    if incar.get('LDAU', False) is False:
        for ldau_flag in ldau_flags:
            if ldau_flag not in skip:
                incar.pop(ldau_flag, None)