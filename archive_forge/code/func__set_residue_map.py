from __future__ import annotations
import os
import tempfile
from shutil import which
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.tempfile import ScratchDir
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.util.coord import get_angle
def _set_residue_map(self) -> None:
    """Map each residue to the corresponding molecule."""
    self.map_residue_to_mol = {}
    lookup = {}
    for idx, mol in enumerate(self.mols):
        if mol.formula not in lookup:
            mol.translate_sites(indices=range(len(mol)), vector=-mol.center_of_mass)
            lookup[mol.formula] = mol.copy()
        self.map_residue_to_mol[f'ml{idx + 1}'] = lookup[mol.formula]