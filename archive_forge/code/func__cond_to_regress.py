from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
def _cond_to_regress(self, info, nscans):
    """Converts condition information to full regressors"""
    reg = []
    regnames = []
    for i, cond in enumerate(info.conditions):
        if hasattr(info, 'amplitudes') and info.amplitudes:
            amplitudes = info.amplitudes[i]
        else:
            amplitudes = None
        regnames.insert(len(regnames), cond)
        scaled_onsets = scale_timings(info.onsets[i], self.inputs.input_units, 'secs', self.inputs.time_repetition)
        scaled_durations = scale_timings(info.durations[i], self.inputs.input_units, 'secs', self.inputs.time_repetition)
        regressor = self._gen_regress(scaled_onsets, scaled_durations, amplitudes, nscans)
        if isdefined(self.inputs.use_temporal_deriv) and self.inputs.use_temporal_deriv:
            reg.insert(len(reg), regressor[0])
            regnames.insert(len(regnames), cond + '_D')
            reg.insert(len(reg), regressor[1])
        else:
            reg.insert(len(reg), regressor)
    nvol = self.inputs.volumes_in_cluster
    if nvol > 1:
        for i in range(nvol - 1):
            treg = np.zeros((nscans / nvol, nvol))
            treg[:, i] = 1
            reg.insert(len(reg), treg.ravel().tolist())
            regnames.insert(len(regnames), 'T1effect_%d' % i)
    return (reg, regnames)