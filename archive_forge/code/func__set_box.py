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
def _set_box(self) -> None:
    """Set the box size for the molecular assembly."""
    net_volume = 0.0
    for idx, mol in enumerate(self.mols):
        length = max((np.max(mol.cart_coords[:, i]) - np.min(mol.cart_coords[:, i]) for i in range(3))) + 2.0
        net_volume += length ** 3.0 * float(self.param_list[idx]['number'])
    length = net_volume ** (1 / 3)
    for idx, _mol in enumerate(self.mols):
        self.param_list[idx]['inside box'] = f'0.0 0.0 0.0 {length} {length} {length}'