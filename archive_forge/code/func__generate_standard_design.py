from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
def _generate_standard_design(self, infolist, functional_runs=None, realignment_parameters=None, outliers=None):
    """Generate a standard design matrix paradigm given information about each run."""
    sessinfo = []
    output_units = 'secs'
    if 'output_units' in self.inputs.traits():
        output_units = self.inputs.output_units
    for i, info in enumerate(infolist):
        sessinfo.insert(i, dict(cond=[]))
        if isdefined(self.inputs.high_pass_filter_cutoff):
            sessinfo[i]['hpf'] = float(self.inputs.high_pass_filter_cutoff)
        if hasattr(info, 'conditions') and info.conditions is not None:
            for cid, cond in enumerate(info.conditions):
                sessinfo[i]['cond'].insert(cid, dict())
                sessinfo[i]['cond'][cid]['name'] = info.conditions[cid]
                scaled_onset = scale_timings(info.onsets[cid], self.inputs.input_units, output_units, self.inputs.time_repetition)
                sessinfo[i]['cond'][cid]['onset'] = scaled_onset
                scaled_duration = scale_timings(info.durations[cid], self.inputs.input_units, output_units, self.inputs.time_repetition)
                sessinfo[i]['cond'][cid]['duration'] = scaled_duration
                if hasattr(info, 'amplitudes') and info.amplitudes:
                    sessinfo[i]['cond'][cid]['amplitudes'] = info.amplitudes[cid]
                if hasattr(info, 'tmod') and info.tmod and (len(info.tmod) > cid):
                    sessinfo[i]['cond'][cid]['tmod'] = info.tmod[cid]
                if hasattr(info, 'pmod') and info.pmod and (len(info.pmod) > cid):
                    if info.pmod[cid]:
                        sessinfo[i]['cond'][cid]['pmod'] = []
                        for j, name in enumerate(info.pmod[cid].name):
                            sessinfo[i]['cond'][cid]['pmod'].insert(j, {})
                            sessinfo[i]['cond'][cid]['pmod'][j]['name'] = name
                            sessinfo[i]['cond'][cid]['pmod'][j]['poly'] = info.pmod[cid].poly[j]
                            sessinfo[i]['cond'][cid]['pmod'][j]['param'] = info.pmod[cid].param[j]
        sessinfo[i]['regress'] = []
        if hasattr(info, 'regressors') and info.regressors is not None:
            for j, r in enumerate(info.regressors):
                sessinfo[i]['regress'].insert(j, dict(name='', val=[]))
                if hasattr(info, 'regressor_names') and info.regressor_names is not None:
                    sessinfo[i]['regress'][j]['name'] = info.regressor_names[j]
                else:
                    sessinfo[i]['regress'][j]['name'] = 'UR%d' % (j + 1)
                sessinfo[i]['regress'][j]['val'] = info.regressors[j]
        sessinfo[i]['scans'] = functional_runs[i]
    if realignment_parameters is not None:
        for i, rp in enumerate(realignment_parameters):
            mc = realignment_parameters[i]
            for col in range(mc.shape[1]):
                colidx = len(sessinfo[i]['regress'])
                sessinfo[i]['regress'].insert(colidx, dict(name='', val=[]))
                sessinfo[i]['regress'][colidx]['name'] = 'Realign%d' % (col + 1)
                sessinfo[i]['regress'][colidx]['val'] = mc[:, col].tolist()
    if outliers is not None:
        for i, out in enumerate(outliers):
            numscans = 0
            for f in ensure_list(sessinfo[i]['scans']):
                shape = load(f).shape
                if len(shape) == 3 or shape[3] == 1:
                    iflogger.warning('You are using 3D instead of 4D files. Are you sure this was intended?')
                    numscans += 1
                else:
                    numscans += shape[3]
            for j, scanno in enumerate(out):
                colidx = len(sessinfo[i]['regress'])
                sessinfo[i]['regress'].insert(colidx, dict(name='', val=[]))
                sessinfo[i]['regress'][colidx]['name'] = 'Outlier%d' % (j + 1)
                sessinfo[i]['regress'][colidx]['val'] = np.zeros((1, numscans))[0].tolist()
                sessinfo[i]['regress'][colidx]['val'][int(scanno)] = 1
    return sessinfo