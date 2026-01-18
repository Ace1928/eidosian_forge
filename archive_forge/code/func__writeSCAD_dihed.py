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
def _writeSCAD_dihed(fp: TextIO, d: 'Dihedron', hedraNdx: Dict, hedraSet: Set[EKT]) -> None:
    fp.write('[ {:9.5f}, {}, {}, {}, '.format(d.angle, hedraNdx[d.h1key], hedraNdx[d.h2key], 1 if d.reverse else 0))
    fp.write(f'{(0 if d.h1key in hedraSet else 1)}, {(0 if d.h2key in hedraSet else 1)}, ')
    fp.write('    // {} [ {} -- {} ] {}\n'.format(d.id, d.hedron1.id, d.hedron2.id, 'reversed' if d.reverse else ''))
    fp.write('        ')
    IC_Chain._write_mtx(fp, d.rcst)
    fp.write(' ]')