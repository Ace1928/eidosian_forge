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
def _add_monomer(self, monomer: Molecule, mon_vector: ArrayLike, move_direction: ArrayLike) -> None:
    """
        extend the polymer molecule by adding a monomer along mon_vector direction.

        Args:
            monomer (Molecule): monomer molecule
            mon_vector (numpy.array): monomer vector that points from head to tail.
            move_direction (numpy.array): direction along which the monomer
                will be positioned
        """
    translate_by = self.molecule.cart_coords[self.end] + self.link_distance * move_direction
    monomer.translate_sites(range(len(monomer)), translate_by)
    if not self.linear_chain:
        self._align_monomer(monomer, mon_vector, move_direction)
    does_cross = False
    for idx, site in enumerate(monomer):
        try:
            self.molecule.append(site.specie, site.coords, properties=site.properties)
        except Exception:
            does_cross = True
            polymer_length = len(self.molecule)
            self.molecule.remove_sites(range(polymer_length - idx, polymer_length))
            break
    if not does_cross:
        self.length += 1
        self.end += len(self.monomer)