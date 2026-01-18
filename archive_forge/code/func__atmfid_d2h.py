import re
from itertools import zip_longest
import numpy as np
from Bio.PDB.PDBExceptions import PDBException
from io import StringIO
from Bio.File import as_handle
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.internal_coords import IC_Residue
from Bio.PDB.PICIO import write_PIC, read_PIC, enumerate_atoms, pdb_date
from typing import Dict, Union, Any, Tuple
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
def _atmfid_d2h(atm: Atom) -> Tuple:
    afid = list(atm.get_full_id())
    afid4 = list(afid[4])
    afid40 = re.sub('D', 'H', afid4[0], count=1)
    new_afid = (afid[0], afid[1], afid[2], afid[3], (afid40, afid4[1]))
    return tuple(new_afid)