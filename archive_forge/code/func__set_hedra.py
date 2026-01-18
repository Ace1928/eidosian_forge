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
def _set_hedra(self) -> Tuple[bool, Hedron, Hedron]:
    """Work out hedra keys and set rev flag."""
    try:
        return (self.rev, self.hedron1, self.hedron2)
    except AttributeError:
        pass
    rev = False
    res = self.ric
    h1key = self.id3
    hedron1 = Dihedron._get_hedron(res, h1key)
    if not hedron1:
        rev = True
        h1key = cast(HKT, tuple(self.atomkeys[2::-1]))
        hedron1 = Dihedron._get_hedron(res, h1key)
        h2key = cast(HKT, tuple(self.atomkeys[3:0:-1]))
    else:
        h2key = self.id32
    if not hedron1:
        raise HedronMatchError(f"can't find 1st hedron for key {h1key} dihedron {self}")
    hedron2 = Dihedron._get_hedron(res, h2key)
    if not hedron2:
        raise HedronMatchError(f"can't find 2nd hedron for key {h2key} dihedron {self}")
    self.hedron1 = hedron1
    self.h1key = h1key
    self.hedron2 = hedron2
    self.h2key = h2key
    self.reverse = rev
    return (rev, hedron1, hedron2)