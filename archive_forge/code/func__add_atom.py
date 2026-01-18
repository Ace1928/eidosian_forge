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
def _add_atom(self, atm: Atom) -> None:
    """Filter Biopython Atom with accept_atoms; set ak_set.

        Arbitrarily renames O' and O'' to O and OXT
        """
    if 'O' == atm.name[0]:
        if "O'" == atm.name:
            atm.name = 'O'
        elif "O''" == atm.name:
            atm.name = 'OXT'
    if atm.name not in self.accept_atoms:
        return
    ak = self.rak(atm)
    self.ak_set.add(ak)