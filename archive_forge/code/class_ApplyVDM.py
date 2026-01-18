import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class ApplyVDM(SPMCommand):
    """Use the fieldmap toolbox from spm to apply the voxel displacement map (VDM) to some epi files.

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=173

    .. important::

        This interface does not deal with real/imag magnitude images nor
        with the two phase files case.

    """
    input_spec = ApplyVDMInputSpec
    output_spec = ApplyVDMOutputSpec
    _jobtype = 'tools'
    _jobname = 'fieldmap'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'in_files':
            return scans_for_fnames(ensure_list(val), keep4d=False, separate_sessions=False)
        if opt == 'vdmfile':
            return scans_for_fname(ensure_list(val))
        return super(ApplyVDM, self)._format_arg(opt, spec, val)

    def _parse_inputs(self):
        """validate spm fieldmap options if set to None ignore"""
        einputs = super(ApplyVDM, self)._parse_inputs()
        return [{'applyvdm': einputs[0]}]

    def _list_outputs(self):
        outputs = self._outputs().get()
        resliced_all = self.inputs.write_which[0] > 0
        resliced_mean = self.inputs.write_which[1] > 0
        if resliced_mean:
            if isinstance(self.inputs.in_files[0], list):
                first_image = self.inputs.in_files[0][0]
            else:
                first_image = self.inputs.in_files[0]
            outputs['mean_image'] = fname_presuffix(first_image, prefix='meanu')
        if resliced_all:
            outputs['out_files'] = []
            for idx, imgf in enumerate(ensure_list(self.inputs.in_files)):
                appliedvdm_run = []
                if isinstance(imgf, list):
                    for i, inner_imgf in enumerate(ensure_list(imgf)):
                        newfile = fname_presuffix(inner_imgf, prefix=self.inputs.out_prefix)
                        appliedvdm_run.append(newfile)
                else:
                    appliedvdm_run = fname_presuffix(imgf, prefix=self.inputs.out_prefix)
                outputs['out_files'].append(appliedvdm_run)
        return outputs