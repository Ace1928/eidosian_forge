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
class LobsterSet(DictSet):
    """Input set to prepare VASP runs that can be digested by Lobster (See cohp.de).

    Args:
        structure (Structure): input structure.
        isym (int): ISYM entry for INCAR, only isym=-1 and isym=0 are allowed
        ismear (int): ISMEAR entry for INCAR, only ismear=-5 and ismear=0 are allowed
        reciprocal_density (int): density of k-mesh by reciprocal volume
        user_supplied_basis (dict): dict including basis functions for all elements in
            structure, e.g. {"Fe": "3d 3p 4s", "O": "2s 2p"}; if not supplied, a
            standard basis is used
        address_basis_file (str): address to a file similar to
            "BASIS_PBE_54_standard.yaml" in pymatgen.io.lobster.lobster_basis
        user_potcar_settings (dict): dict including potcar settings for all elements in
            structure, e.g. {"Fe": "Fe_pv", "O": "O"}; if not supplied, a standard basis
            is used.
        **kwargs: Other kwargs supported by DictSet.
    """
    isym: int = 0
    ismear: int = -5
    reciprocal_density: int | None = None
    address_basis_file: str | None = None
    user_supplied_basis: dict | None = None
    user_potcar_functional: UserPotcarFunctional = 'PBE_54'
    CONFIG = MPRelaxSet.CONFIG
    _valid_potcars = ('PBE_52', 'PBE_54')

    def __post_init__(self):
        super().__post_init__()
        warnings.warn('Make sure that all parameters are okay! This is a brand new implementation.')
        if self.isym not in (-1, 0):
            raise ValueError('Lobster cannot digest WAVEFUNCTIONS with symmetry. isym must be -1 or 0')
        if self.ismear not in (-5, 0):
            raise ValueError('Lobster usually works with ismear=-5 or ismear=0')
        self._config_dict['POTCAR']['W'] = 'W_sv'

    @property
    def kpoints_updates(self) -> dict | Kpoints:
        """Get updates to the kpoints configuration for this calculation type."""
        return {'reciprocal_density': self.reciprocal_density or 310}

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        from pymatgen.io.lobster import Lobsterin
        potcar_symbols = self.poscar.site_symbols
        if self.user_supplied_basis is None and self.address_basis_file is None:
            basis = Lobsterin.get_basis(structure=self.structure, potcar_symbols=potcar_symbols)
        elif self.address_basis_file is not None:
            basis = Lobsterin.get_basis(structure=self.structure, potcar_symbols=potcar_symbols, address_basis_file=self.address_basis_file)
        elif self.user_supplied_basis is not None:
            for atom_type in self.structure.symbol_set:
                if atom_type not in self.user_supplied_basis:
                    raise ValueError(f'There are no basis functions for the atom type {atom_type}')
            basis = [f'{key} {value}' for key, value in self.user_supplied_basis.items()]
        lobsterin = Lobsterin(settingsdict={'basisfunctions': basis})
        nbands = lobsterin._get_nbands(structure=self.structure)
        return {'EDIFF': 1e-06, 'NSW': 0, 'LWAVE': True, 'ISYM': self.isym, 'NBANDS': nbands, 'IBRION': -1, 'ISMEAR': self.ismear, 'LORBIT': 11, 'ICHARG': 0, 'ALGO': 'Normal'}