import re
from collections import deque, namedtuple
import copy
from numbers import Integral
import numpy as np  # type: ignore
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.vectors import multi_coord_space, multi_rot_Z
from Bio.PDB.vectors import coord_space
from Bio.PDB.ic_data import ic_data_backbone, ic_data_sidechains
from Bio.PDB.ic_data import primary_angles
from Bio.PDB.ic_data import ic_data_sidechain_extras, residue_atom_bond_state
from Bio.PDB.ic_data import dihedra_primary_defaults, hedra_defaults
from typing import (
def _do_bond_rotate(self, base: 'Dihedron', delta: float):
    """Find and modify related dihedra through id3_dh_index."""
    try:
        for dk in self.cic.id3_dh_index[base.id3]:
            dihed = self.cic.dihedra[dk]
            dihed.angle += delta
            try:
                for d2rk in self.cic.id3_dh_index[dihed.id32[::-1]]:
                    self.cic.dihedra[d2rk].angle += delta
            except KeyError:
                pass
    except AttributeError:
        raise RuntimeError('bond_rotate, bond_set only for dihedral angles')