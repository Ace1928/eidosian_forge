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
def _spec_glyCB(self) -> None:
    """Populate values for Gly C-beta."""
    Ca_Cb_Len = 1.53363
    if hasattr(self, 'scale'):
        Ca_Cb_Len *= self.scale
    for gcbd in self.gcb.values():
        cbak = gcbd[3]
        self.atomArrayValid[self.atomArrayIndex[cbak]] = False
        ric = cbak.ric
        rN, rCA, rC, rO = (ric.rak('N'), ric.rak('CA'), ric.rak('C'), ric.rak('O'))
        gCBd = self.dihedra[gcbd]
        dndx = gCBd.ndx
        h2ndx = gCBd.hedron2.ndx
        self.hedraL12[h2ndx] = Ca_Cb_Len
        self.hedraAngle[h2ndx] = 110.17513
        self.hedraL23[h2ndx] = self.hedraL12[self.hedraNdx[rCA, rC, rO]]
        self.hAtoms_needs_update[gCBd.hedron2.ndx] = True
        for ak in gCBd.hedron2.atomkeys:
            self.atomArrayValid[self.atomArrayIndex[ak]] = False
        refval = self.dihedra.get((rN, rCA, rC, rO), None)
        if refval:
            angl = 122.68219 + self.dihedraAngle[refval.ndx]
            self.dihedraAngle[dndx] = angl if angl <= 180.0 else angl - 360.0
        else:
            self.dihedraAngle[dndx] = 120