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
def _gen_edra(self, lst: Union[Tuple['AtomKey', ...], List['AtomKey']]) -> None:
    """Populate hedra/dihedra given edron ID tuple.

        Given list of AtomKeys defining hedron or dihedron
          convert to AtomKeys with coordinates in this residue
          add appropriately to self.di/hedra, expand as needed atom altlocs

        :param list lst: tuple of AtomKeys.
            Specifies Hedron or Dihedron
        """
    for ak in lst:
        if ak.missing:
            return
    lenLst = len(lst)
    if 4 > lenLst:
        cdct, dct, obj = (self.cic.hedra, self.hedra, Hedron)
    else:
        cdct, dct, obj = (self.cic.dihedra, self.dihedra, Dihedron)
    if isinstance(lst, List):
        tlst = tuple(lst)
    else:
        tlst = lst
    hl = self.split_akl(tlst)
    for tnlst in hl:
        if len(tnlst) == lenLst:
            if tnlst not in cdct:
                cdct[tnlst] = obj(tnlst)
            if tnlst not in dct:
                dct[tnlst] = cdct[tnlst]
            dct[tnlst].needs_update = True