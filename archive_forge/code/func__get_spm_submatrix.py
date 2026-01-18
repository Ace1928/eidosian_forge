import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
def _get_spm_submatrix(self, spmmat, sessidx, rows=None):
    """
        Parameters
        ----------
        spmmat: scipy matlab object
            full SPM.mat file loaded into a scipy object
        sessidx: int
            index to session that needs to be extracted.
        """
    designmatrix = spmmat['SPM'][0][0].xX[0][0].X
    U = spmmat['SPM'][0][0].Sess[0][sessidx].U[0]
    if rows is None:
        rows = spmmat['SPM'][0][0].Sess[0][sessidx].row[0] - 1
    cols = spmmat['SPM'][0][0].Sess[0][sessidx].col[0][list(range(len(U)))] - 1
    outmatrix = designmatrix.take(rows.tolist(), axis=0).take(cols.tolist(), axis=1)
    return outmatrix