import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class DARTELNorm2MNI(SPMCommand):
    """Use spm DARTEL to normalize data to MNI space

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=188

    Examples
    --------
    >>> import nipype.interfaces.spm as spm
    >>> nm = spm.DARTELNorm2MNI()
    >>> nm.inputs.template_file = 'Template_6.nii'
    >>> nm.inputs.flowfield_files = ['u_rc1s1_Template.nii', 'u_rc1s3_Template.nii']
    >>> nm.inputs.apply_to_files = ['c1s1.nii', 'c1s3.nii']
    >>> nm.inputs.modulate = True
    >>> nm.run() # doctest: +SKIP

    """
    input_spec = DARTELNorm2MNIInputSpec
    output_spec = DARTELNorm2MNIOutputSpec
    _jobtype = 'tools'
    _jobname = 'dartel'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['template_file']:
            return np.array([val], dtype=object)
        elif opt in ['flowfield_files']:
            return scans_for_fnames(val, keep4d=True)
        elif opt in ['apply_to_files']:
            return scans_for_fnames(val, keep4d=True, separate_sessions=True)
        elif opt == 'voxel_size':
            return list(val)
        elif opt == 'bounding_box':
            return list(val)
        elif opt == 'fwhm':
            if isinstance(val, list):
                return val
            else:
                return [val, val, val]
        else:
            return super(DARTELNorm2MNI, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        pth, base, ext = split_filename(self.inputs.template_file)
        outputs['normalization_parameter_file'] = os.path.realpath(base + '_2mni.mat')
        outputs['normalized_files'] = []
        prefix = 'w'
        if isdefined(self.inputs.modulate) and self.inputs.modulate:
            prefix = 'm' + prefix
        if not isdefined(self.inputs.fwhm) or self.inputs.fwhm > 0:
            prefix = 's' + prefix
        for filename in self.inputs.apply_to_files:
            pth, base, ext = split_filename(filename)
            outputs['normalized_files'].append(os.path.realpath('%s%s%s' % (prefix, base, ext)))
        return outputs