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
def _get_ak_tuple(self, ak_str: str) -> Optional[Tuple['AtomKey', ...]]:
    """Convert atom pair string to AtomKey tuple.

        :param str ak_str:
            Two atom names separated by ':', e.g. 'N:CA'
            Optional position specifier relative to self,
            e.g. '-1C:N' for preceding peptide bond.
        """
    AK = AtomKey
    S = self
    angle_key2 = []
    akstr_list = ak_str.split(':')
    lenInput = len(akstr_list)
    for a in akstr_list:
        m = self._relative_atom_re.match(a)
        if m:
            if m.group(1) == '-1':
                if 0 < len(S.rprev):
                    angle_key2.append(AK(S.rprev[0], m.group(2)))
            elif m.group(1) == '1':
                if 0 < len(S.rnext):
                    angle_key2.append(AK(S.rnext[0], m.group(2)))
            elif m.group(1) == '0':
                angle_key2.append(self.rak(m.group(2)))
        else:
            angle_key2.append(self.rak(a))
    if len(angle_key2) != lenInput:
        return None
    return tuple(angle_key2)