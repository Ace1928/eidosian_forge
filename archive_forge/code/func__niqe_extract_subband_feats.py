from ..utils import *
import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
def _niqe_extract_subband_feats(mscncoefs):
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0, alpha1, N1, bl1, br1, alpha2, N2, bl2, br2, alpha3, N3, bl3, bl3, alpha4, N4, bl4, bl4])