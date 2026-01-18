from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
class SpecifySparseModel(SpecifyModel):
    """Specify a sparse model that is compatible with SPM/FSL designers [1]_.

    Examples
    --------
    >>> from nipype.algorithms import modelgen
    >>> from nipype.interfaces.base import Bunch
    >>> s = modelgen.SpecifySparseModel()
    >>> s.inputs.input_units = 'secs'
    >>> s.inputs.functional_runs = ['functional2.nii', 'functional3.nii']
    >>> s.inputs.time_repetition = 6
    >>> s.inputs.time_acquisition = 2
    >>> s.inputs.high_pass_filter_cutoff = 128.
    >>> s.inputs.model_hrf = True
    >>> evs_run2 = Bunch(conditions=['cond1'], onsets=[[2, 50, 100, 180]],
    ...                  durations=[[1]])
    >>> evs_run3 = Bunch(conditions=['cond1'], onsets=[[30, 40, 100, 150]],
    ...                  durations=[[1]])
    >>> s.inputs.subject_info = [evs_run2, evs_run3]  # doctest: +SKIP

    References
    ----------
    .. [1] Perrachione TK and Ghosh SS (2013) Optimized design and analysis of
       sparse-sampling fMRI experiments. Front. Neurosci. 7:55
       http://journal.frontiersin.org/Journal/10.3389/fnins.2013.00055/abstract

    """
    input_spec = SpecifySparseModelInputSpec
    output_spec = SpecifySparseModelOutputSpec

    def _gen_regress(self, i_onsets, i_durations, i_amplitudes, nscans):
        """Generates a regressor for a sparse/clustered-sparse acquisition"""
        bplot = False
        if isdefined(self.inputs.save_plot) and self.inputs.save_plot:
            bplot = True
            import matplotlib
            matplotlib.use(config.get('execution', 'matplotlib_backend'))
            import matplotlib.pyplot as plt
        TR = int(np.round(self.inputs.time_repetition * 1000))
        if self.inputs.time_acquisition:
            TA = int(np.round(self.inputs.time_acquisition * 1000))
        else:
            TA = TR
        nvol = self.inputs.volumes_in_cluster
        SCANONSET = np.round(self.inputs.scan_onset * 1000)
        total_time = TR * (nscans - nvol) / nvol + TA * nvol + SCANONSET
        SILENCE = TR - TA * nvol
        dt = TA / 10.0
        durations = np.round(np.array(i_durations) * 1000)
        if len(durations) == 1:
            durations = durations * np.ones(len(i_onsets))
        onsets = np.round(np.array(i_onsets) * 1000)
        dttemp = math.gcd(TA, math.gcd(SILENCE, TR))
        if dt < dttemp:
            if dttemp % dt != 0:
                dt = float(math.gcd(dttemp, int(dt)))
        if dt < 1:
            raise Exception('Time multiple less than 1 ms')
        iflogger.info('Setting dt = %d ms\n', dt)
        npts = int(np.ceil(total_time / dt))
        times = np.arange(0, total_time, dt) * 0.001
        timeline = np.zeros(npts)
        timeline2 = np.zeros(npts)
        if isdefined(self.inputs.model_hrf) and self.inputs.model_hrf:
            hrf = spm_hrf(dt * 0.001)
        reg_scale = 1.0
        if self.inputs.scale_regressors:
            boxcar = np.zeros(int(50.0 * 1000.0 / dt))
            if self.inputs.stimuli_as_impulses:
                boxcar[int(1.0 * 1000.0 / dt)] = 1.0
                reg_scale = float(TA / dt)
            else:
                boxcar[int(1.0 * 1000.0 / dt):int(2.0 * 1000.0 / dt)] = 1.0
            if isdefined(self.inputs.model_hrf) and self.inputs.model_hrf:
                response = np.convolve(boxcar, hrf)
                reg_scale = 1.0 / response.max()
                iflogger.info('response sum: %.4f max: %.4f', response.sum(), response.max())
            iflogger.info('reg_scale: %.4f', reg_scale)
        for i, t in enumerate(onsets):
            idx = int(np.round(t / dt))
            if i_amplitudes:
                if len(i_amplitudes) > 1:
                    timeline2[idx] = i_amplitudes[i]
                else:
                    timeline2[idx] = i_amplitudes[0]
            else:
                timeline2[idx] = 1
            if bplot:
                plt.subplot(4, 1, 1)
                plt.plot(times, timeline2)
            if not self.inputs.stimuli_as_impulses:
                if durations[i] == 0:
                    durations[i] = TA * nvol
                stimdur = np.ones(int(durations[i] / dt))
                timeline2 = np.convolve(timeline2, stimdur)[0:len(timeline2)]
            timeline += timeline2
            timeline2[:] = 0
        if bplot:
            plt.subplot(4, 1, 2)
            plt.plot(times, timeline)
        if isdefined(self.inputs.model_hrf) and self.inputs.model_hrf:
            timeline = np.convolve(timeline, hrf)[0:len(timeline)]
            if isdefined(self.inputs.use_temporal_deriv) and self.inputs.use_temporal_deriv:
                timederiv = np.concatenate(([0], np.diff(timeline)))
        if bplot:
            plt.subplot(4, 1, 3)
            plt.plot(times, timeline)
            if isdefined(self.inputs.use_temporal_deriv) and self.inputs.use_temporal_deriv:
                plt.plot(times, timederiv)
        timeline2 = np.zeros(npts)
        reg = []
        regderiv = []
        for i, trial in enumerate(np.arange(nscans) / nvol):
            scanstart = int((SCANONSET + trial * TR + i % nvol * TA) / dt)
            scanidx = scanstart + np.arange(int(TA / dt))
            timeline2[scanidx] = np.max(timeline)
            reg.insert(i, np.mean(timeline[scanidx]) * reg_scale)
            if isdefined(self.inputs.use_temporal_deriv) and self.inputs.use_temporal_deriv:
                regderiv.insert(i, np.mean(timederiv[scanidx]) * reg_scale)
        if isdefined(self.inputs.use_temporal_deriv) and self.inputs.use_temporal_deriv:
            iflogger.info('orthoganlizing derivative w.r.t. main regressor')
            regderiv = orth(reg, regderiv)
        if bplot:
            plt.subplot(4, 1, 3)
            plt.plot(times, timeline2)
            plt.subplot(4, 1, 4)
            plt.bar(np.arange(len(reg)), reg, width=0.5)
            plt.savefig('sparse.png')
            plt.savefig('sparse.svg')
        if regderiv:
            return [reg, regderiv]
        else:
            return reg

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

    def _generate_design(self, infolist=None):
        if isdefined(self.inputs.subject_info):
            infolist = self.inputs.subject_info
        else:
            infolist = gen_info(self.inputs.event_files)
        sparselist = self._generate_clustered_design(infolist)
        super(SpecifySparseModel, self)._generate_design(infolist=sparselist)

    def _list_outputs(self):
        outputs = self._outputs().get()
        if not hasattr(self, '_sessinfo'):
            self._generate_design()
        outputs['session_info'] = self._sessinfo
        if isdefined(self.inputs.save_plot) and self.inputs.save_plot:
            outputs['sparse_png_file'] = os.path.join(os.getcwd(), 'sparse.png')
            outputs['sparse_svg_file'] = os.path.join(os.getcwd(), 'sparse.svg')
        return outputs