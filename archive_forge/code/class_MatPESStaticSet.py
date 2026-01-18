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
class MatPESStaticSet(DictSet):
    """Creates input files for a MatPES static calculation.

    The goal of MatPES is to generate potential energy surface data. This is a distinctly different
    from the objectives of the MP static calculations, which aims to obtain primarily accurate
    energies and also electronic structure (DOS). For PES data, force accuracy (and to some extent,
    stress accuracy) is of paramount importance.

    The default POTCAR versions have been updated to PBE_54 from the old PBE set used in the
    MPStaticSet. However, **U values** are still based on PBE. The implicit assumption here is that
    the PBE_54 and PBE POTCARs are sufficiently similar that the U values fitted to the old PBE
    functional still applies.

    Args:
        structure (Structure): The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set separately before
            the inputs are generated.
        xc_functional ('R2SCAN'|'PBE'): Exchange-correlation functional to use. Defaults to 'PBE'.
        **kwargs: Same as those supported by DictSet.
    """
    xc_functional: Literal['R2SCAN', 'PBE', 'PBE+U'] = 'PBE'
    prev_incar: dict | str | None = None
    inherit_incar: list[str] | bool = ('LPEAD', 'NGX', 'NGY', 'NGZ', 'SYMPREC', 'IMIX', 'LMAXMIX', 'KGAMMA', 'ISYM', 'NCORE', 'NPAR', 'NELMIN', 'IOPT', 'NBANDS', 'KPAR', 'AMIN', 'NELMDL', 'BMIX', 'AMIX_MAG', 'BMIX_MAG')
    CONFIG = _load_yaml_config('MatPESStaticSet')

    def __post_init__(self):
        """Validate inputs"""
        super().__post_init__()
        valid_xc_functionals = ('R2SCAN', 'PBE', 'PBE+U')
        if self.xc_functional.upper() not in valid_xc_functionals:
            raise ValueError(f"Unrecognized xc_functional='{self.xc_functional}'. Supported exchange-correlation functionals are {valid_xc_functionals}")
        default_potcars = self.CONFIG['PARENT'].replace('PBE', 'PBE_').replace('BASE', '')
        self.user_potcar_functional = self.user_potcar_functional or default_potcars
        if self.user_potcar_functional.upper() != default_potcars:
            warnings.warn(f'self.user_potcar_functional={self.user_potcar_functional!r} is inconsistent with the recommended {default_potcars}.', UserWarning)

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        updates: dict[str, Any] = {}
        if self.xc_functional.upper() == 'R2SCAN':
            updates.update({'METAGGA': 'R2SCAN', 'ALGO': 'ALL', 'GGA': None})
        if self.xc_functional.upper().endswith('+U'):
            updates['LDAU'] = True
        return updates