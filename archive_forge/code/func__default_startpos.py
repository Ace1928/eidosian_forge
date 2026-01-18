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
def _default_startpos(self) -> Dict['AtomKey', np.array]:
    """Generate default N-Ca-C coordinates to build this residue from."""
    atomCoords = {}
    cic = self.cic
    dlist0 = [cic.id3_dh_index.get(akl, None) for akl in sorted(self.NCaCKey)]
    dlist1 = [d for d in dlist0 if d is not None]
    dlist = [cic.dihedra[val] for sublist in dlist1 for val in sublist]
    for d in dlist:
        for i, a in enumerate(d.atomkeys):
            atomCoords[a] = cic.dAtoms[d.ndx][i]
    return atomCoords