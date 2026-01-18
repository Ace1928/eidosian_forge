import os
from functools import lru_cache
import numpy as np
from numpy import ones, kron, mean, eye, hstack, tile
from numpy.linalg import pinv
import nibabel as nb
from ..interfaces.base import (
def ICC_rep_anova(Y, projection_matrix=None):
    """
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns

    One Sample Repeated measure ANOVA

    Y = XB + E with X = [FaTor / Subjects]

    ``ICC_rep_anova`` involves an expensive operation to compute a projection
    matrix, which depends only on the shape of ``Y``, which is computed by
    calling ``ICC_projection_matrix(Y.shape)``. If arrays of multiple shapes are
    expected, it may be worth pre-computing and passing directly as an
    argument to ``ICC_rep_anova``.

    If only one ``Y.shape`` will occur, you do not need to explicitly handle
    these, as the most recently calculated matrix is cached automatically.
    For example, if you are running the same computation on every voxel of
    an image, you will see significant speedups.

    If a ``Y`` is passed with a new shape, a new matrix will be calculated
    automatically.
    """
    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfr = nb_subjects - 1
    dfe = dfr * dfc
    demeaned_Y = Y - mean(Y)
    SST = np.sum(demeaned_Y ** 2)
    if projection_matrix is None:
        projection_matrix = ICC_projection_matrix(Y.shape)
    residuals = Y.flatten('F') - projection_matrix @ Y.flatten('F')
    SSE = np.sum(residuals ** 2)
    MSE = SSE / dfe
    SSC = np.sum(mean(demeaned_Y, 0) ** 2) * nb_subjects
    MSC = SSC / dfc / nb_subjects
    session_effect_F = MSC / MSE
    SSR = SST - SSC - SSE
    MSR = SSR / dfr
    ICC = (MSR - MSE) / (MSR + dfc * MSE)
    e_var = MSE
    r_var = (MSR - MSE) / nb_conditions
    return (ICC, r_var, e_var, session_effect_F, dfc, dfe)