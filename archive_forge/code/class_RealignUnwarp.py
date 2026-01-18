import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class RealignUnwarp(SPMCommand):
    """Use spm_uw_estimate for estimating within subject registration and unwarping
    of time series. Function accepts only one single field map. If in_files is a
    list of files they will be treated as separate sessions but associated to the
    same fieldmap.

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=31

    Examples
    --------

    >>> import nipype.interfaces.spm as spm
    >>> realignUnwarp = spm.RealignUnwarp()
    >>> realignUnwarp.inputs.in_files = ['functional.nii', 'functional2.nii']
    >>> realignUnwarp.inputs.phase_map = 'voxeldisplacemap.vdm'
    >>> realignUnwarp.inputs.register_to_mean = True
    >>> realignUnwarp.run() # doctest: +SKIP

    """
    input_spec = RealignUnwarpInputSpec
    output_spec = RealignUnwarpOutputSpec
    _jobtype = 'spatial'
    _jobname = 'realignunwarp'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'in_files':
            return scans_for_fnames(ensure_list(val), keep4d=False, separate_sessions=True)
        return super(RealignUnwarp, self)._format_arg(opt, spec, val)

    def _parse_inputs(self, skip=()):
        spmdict = super(RealignUnwarp, self)._parse_inputs(skip=())[0]
        if isdefined(self.inputs.phase_map):
            pmscan = spmdict['data']['pmscan']
        else:
            pmscan = ''
        if isdefined(self.inputs.in_files):
            if isinstance(self.inputs.in_files, list):
                data = [dict(scans=sess, pmscan=pmscan) for sess in spmdict['data']['scans']]
            else:
                data = [dict(scans=spmdict['data']['scans'], pmscan=pmscan)]
        spmdict['data'] = data
        return [spmdict]

    def _list_outputs(self):
        outputs = self._outputs().get()
        resliced_all = self.inputs.reslice_which[0] > 0
        resliced_mean = self.inputs.reslice_which[1] > 0
        if isdefined(self.inputs.in_files):
            outputs['realignment_parameters'] = []
        for imgf in self.inputs.in_files:
            if isinstance(imgf, list):
                tmp_imgf = imgf[0]
            else:
                tmp_imgf = imgf
            outputs['realignment_parameters'].append(fname_presuffix(tmp_imgf, prefix='rp_', suffix='.txt', use_ext=False))
            if not isinstance(imgf, list) and func_is_3d(imgf):
                break
        if isinstance(self.inputs.in_files[0], list):
            first_image = self.inputs.in_files[0][0]
        else:
            first_image = self.inputs.in_files[0]
        if resliced_mean:
            outputs['mean_image'] = fname_presuffix(first_image, prefix='meanu')
        if resliced_all:
            outputs['realigned_unwarped_files'] = []
            for idx, imgf in enumerate(ensure_list(self.inputs.in_files)):
                realigned_run = []
                if isinstance(imgf, list):
                    for i, inner_imgf in enumerate(ensure_list(imgf)):
                        newfile = fname_presuffix(inner_imgf, prefix=self.inputs.out_prefix)
                        realigned_run.append(newfile)
                else:
                    realigned_run = fname_presuffix(imgf, prefix=self.inputs.out_prefix)
                outputs['realigned_unwarped_files'].append(realigned_run)
        return outputs