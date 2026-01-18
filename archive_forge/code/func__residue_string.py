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
def _residue_string(res: 'Residue') -> str:
    """Generate PIC Residue string.

        Enough to create Biopython Residue object without actual Atoms.

        :param Residue res: Biopython Residue object reference
        """
    segid = res.get_segid()
    if segid.isspace() or '' == segid:
        segid = ''
    else:
        segid = ' [' + segid + ']'
    return str(res.get_full_id()) + ' ' + res.resname + segid + '\n'