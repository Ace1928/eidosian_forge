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
@staticmethod
def _get_hedron(ic_res: IC_Residue, id3: HKT) -> Optional[Hedron]:
    """Find specified hedron on this residue or its adjacent neighbors."""
    hedron = ic_res.hedra.get(id3, None)
    if not hedron and 0 < len(ic_res.rprev):
        for rp in ic_res.rprev:
            hedron = rp.hedra.get(id3, None)
            if hedron is not None:
                break
    if not hedron and 0 < len(ic_res.rnext):
        for rn in ic_res.rnext:
            hedron = rn.hedra.get(id3, None)
            if hedron is not None:
                break
    return hedron