from Bio.File import as_handle
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.internal_coords import IC_Residue, IC_Chain
from Bio.PDB.vectors import homog_scale_mtx
import numpy as np  # type: ignore
def _scale_residue(res, scale, scaleMtx):
    if res.internal_coord:
        res.internal_coord.applyMtx(scaleMtx)
        if res.internal_coord.gly_Cbeta:
            res.internal_coord.scale = scale