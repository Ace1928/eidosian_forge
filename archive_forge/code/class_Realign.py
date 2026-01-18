import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class Realign(SPMCommand):
    """Use spm_realign for estimating within modality rigid body alignment

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=25

    Examples
    --------

    >>> import nipype.interfaces.spm as spm
    >>> realign = spm.Realign()
    >>> realign.inputs.in_files = 'functional.nii'
    >>> realign.inputs.register_to_mean = True
    >>> realign.run() # doctest: +SKIP

    """
    input_spec = RealignInputSpec
    output_spec = RealignOutputSpec
    _jobtype = 'spatial'
    _jobname = 'realign'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'in_files':
            if self.inputs.jobtype == 'write':
                separate_sessions = False
            else:
                separate_sessions = True
            return scans_for_fnames(val, keep4d=False, separate_sessions=separate_sessions)
        return super(Realign, self)._format_arg(opt, spec, val)

    def _parse_inputs(self):
        """validate spm realign options if set to None ignore"""
        einputs = super(Realign, self)._parse_inputs()
        return [{'%s' % self.inputs.jobtype: einputs[0]}]

    def _list_outputs(self):
        outputs = self._outputs().get()
        resliced_all = self.inputs.write_which[0] > 0
        resliced_mean = self.inputs.write_which[1] > 0
        if self.inputs.jobtype != 'write':
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
        if self.inputs.jobtype == 'estimate':
            outputs['realigned_files'] = self.inputs.in_files
        if self.inputs.jobtype == 'estimate' or self.inputs.jobtype == 'estwrite':
            outputs['modified_in_files'] = self.inputs.in_files
        if self.inputs.jobtype == 'write' or self.inputs.jobtype == 'estwrite':
            if isinstance(self.inputs.in_files[0], list):
                first_image = self.inputs.in_files[0][0]
            else:
                first_image = self.inputs.in_files[0]
            if resliced_mean:
                outputs['mean_image'] = fname_presuffix(first_image, prefix='mean')
            if resliced_all:
                outputs['realigned_files'] = []
                for idx, imgf in enumerate(ensure_list(self.inputs.in_files)):
                    realigned_run = []
                    if isinstance(imgf, list):
                        for i, inner_imgf in enumerate(ensure_list(imgf)):
                            newfile = fname_presuffix(inner_imgf, prefix=self.inputs.out_prefix)
                            realigned_run.append(newfile)
                    else:
                        realigned_run = fname_presuffix(imgf, prefix=self.inputs.out_prefix)
                    outputs['realigned_files'].append(realigned_run)
        return outputs