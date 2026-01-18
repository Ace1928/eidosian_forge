import numpy as np
import warnings
from Bio.File import as_handle
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
def _mmcif_get(self, key, dict, deflt):
    if key in dict:
        rslt = dict[key][0]
        if '?' != rslt:
            return rslt
    return deflt