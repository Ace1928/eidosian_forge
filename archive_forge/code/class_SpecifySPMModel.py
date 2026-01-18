from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
class SpecifySPMModel(SpecifyModel):
    """Add SPM specific options to SpecifyModel

    Adds:

       - concatenate_runs
       - output_units

    Examples
    --------
    >>> from nipype.algorithms import modelgen
    >>> from nipype.interfaces.base import Bunch
    >>> s = modelgen.SpecifySPMModel()
    >>> s.inputs.input_units = 'secs'
    >>> s.inputs.output_units = 'scans'
    >>> s.inputs.high_pass_filter_cutoff = 128.
    >>> s.inputs.functional_runs = ['functional2.nii', 'functional3.nii']
    >>> s.inputs.time_repetition = 6
    >>> s.inputs.concatenate_runs = True
    >>> evs_run2 = Bunch(conditions=['cond1'], onsets=[[2, 50, 100, 180]], durations=[[1]])
    >>> evs_run3 = Bunch(conditions=['cond1'], onsets=[[30, 40, 100, 150]], durations=[[1]])
    >>> s.inputs.subject_info = [evs_run2, evs_run3]

    """
    input_spec = SpecifySPMModelInputSpec

    def _concatenate_info(self, infolist):
        nscans = []
        for i, f in enumerate(self.inputs.functional_runs):
            if isinstance(f, list):
                numscans = len(f)
            elif isinstance(f, (str, bytes)):
                img = load(f)
                numscans = img.shape[3]
            else:
                raise Exception('Functional input not specified correctly')
            nscans.insert(i, numscans)
        infoout = infolist[0]
        for j, val in enumerate(infolist[0].durations):
            if len(infolist[0].onsets[j]) > 1 and len(val) == 1:
                infoout.durations[j] = infolist[0].durations[j] * len(infolist[0].onsets[j])
        for i, info in enumerate(infolist[1:]):
            if info.onsets:
                for j, val in enumerate(info.onsets):
                    if self.inputs.input_units == 'secs':
                        onsets = np.array(info.onsets[j]) + self.inputs.time_repetition * sum(nscans[0:i + 1])
                        infoout.onsets[j].extend(onsets.tolist())
                    else:
                        onsets = np.array(info.onsets[j]) + sum(nscans[0:i + 1])
                        infoout.onsets[j].extend(onsets.tolist())
                for j, val in enumerate(info.durations):
                    if len(info.onsets[j]) > 1 and len(val) == 1:
                        infoout.durations[j].extend(info.durations[j] * len(info.onsets[j]))
                    elif len(info.onsets[j]) == len(val):
                        infoout.durations[j].extend(info.durations[j])
                    else:
                        raise ValueError('Mismatch in number of onsets and                                           durations for run {0}, condition                                           {1}'.format(i + 2, j + 1))
                if hasattr(info, 'amplitudes') and info.amplitudes:
                    for j, val in enumerate(info.amplitudes):
                        infoout.amplitudes[j].extend(info.amplitudes[j])
                if hasattr(info, 'pmod') and info.pmod:
                    for j, val in enumerate(info.pmod):
                        if val:
                            for key, data in enumerate(val.param):
                                infoout.pmod[j].param[key].extend(data)
            if hasattr(info, 'regressors') and info.regressors:
                for j, v in enumerate(info.regressors):
                    infoout.regressors[j].extend(info.regressors[j])
            if not hasattr(infoout, 'regressors') or not infoout.regressors:
                infoout.regressors = []
            onelist = np.zeros((1, sum(nscans)))
            onelist[0, sum(nscans[0:i]):sum(nscans[0:i + 1])] = 1
            infoout.regressors.insert(len(infoout.regressors), onelist.tolist()[0])
        return ([infoout], nscans)

    def _generate_design(self, infolist=None):
        if not isdefined(self.inputs.concatenate_runs) or not self.inputs.concatenate_runs:
            super(SpecifySPMModel, self)._generate_design(infolist=infolist)
            return
        if isdefined(self.inputs.subject_info):
            infolist = self.inputs.subject_info
        else:
            infolist = gen_info(self.inputs.event_files)
        concatlist, nscans = self._concatenate_info(infolist)
        functional_runs = [ensure_list(self.inputs.functional_runs)]
        realignment_parameters = []
        if isdefined(self.inputs.realignment_parameters):
            realignment_parameters = []
            for parfile in self.inputs.realignment_parameters:
                mc = np.apply_along_axis(func1d=normalize_mc_params, axis=1, arr=np.loadtxt(parfile), source=self.inputs.parameter_source)
                if not realignment_parameters:
                    realignment_parameters.insert(0, mc)
                else:
                    realignment_parameters[0] = np.concatenate((realignment_parameters[0], mc))
        outliers = []
        if isdefined(self.inputs.outlier_files):
            outliers = [[]]
            for i, filename in enumerate(self.inputs.outlier_files):
                try:
                    out = np.loadtxt(filename)
                except IOError:
                    iflogger.warning('Error reading outliers file %s', filename)
                    out = np.array([])
                if out.size > 0:
                    iflogger.debug('fname=%s, out=%s, nscans=%d', filename, out, sum(nscans[0:i]))
                    sumscans = out.astype(int) + sum(nscans[0:i])
                    if out.size == 1:
                        outliers[0] += [np.array(sumscans, dtype=int).tolist()]
                    else:
                        outliers[0] += np.array(sumscans, dtype=int).tolist()
        self._sessinfo = self._generate_standard_design(concatlist, functional_runs=functional_runs, realignment_parameters=realignment_parameters, outliers=outliers)