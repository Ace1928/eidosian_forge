from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
def _generate_clustered_design(self, infolist):
    """Generates condition information for sparse-clustered
        designs.

        """
    infoout = deepcopy(infolist)
    for i, info in enumerate(infolist):
        infoout[i].conditions = None
        infoout[i].onsets = None
        infoout[i].durations = None
        if info.conditions:
            img = load(self.inputs.functional_runs[i])
            nscans = img.shape[3]
            reg, regnames = self._cond_to_regress(info, nscans)
            if hasattr(infoout[i], 'regressors') and infoout[i].regressors:
                if not infoout[i].regressor_names:
                    infoout[i].regressor_names = ['R%d' % j for j in range(len(infoout[i].regressors))]
            else:
                infoout[i].regressors = []
                infoout[i].regressor_names = []
            for j, r in enumerate(reg):
                regidx = len(infoout[i].regressors)
                infoout[i].regressor_names.insert(regidx, regnames[j])
                infoout[i].regressors.insert(regidx, r)
    return infoout