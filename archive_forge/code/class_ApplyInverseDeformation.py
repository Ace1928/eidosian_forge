import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class ApplyInverseDeformation(SPMCommand):
    """Uses spm to apply inverse deformation stored in a .mat file or a
    deformation field to a given file

    Examples
    --------

    >>> import nipype.interfaces.spm.utils as spmu
    >>> inv = spmu.ApplyInverseDeformation()
    >>> inv.inputs.in_files = 'functional.nii'
    >>> inv.inputs.deformation = 'struct_to_func.mat'
    >>> inv.inputs.target = 'structural.nii'
    >>> inv.run() # doctest: +SKIP
    """
    input_spec = ApplyInverseDeformationInput
    output_spec = ApplyInverseDeformationOutput
    _jobtype = 'util'
    _jobname = 'defs'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'in_files':
            return scans_for_fnames(ensure_list(val))
        if opt == 'target':
            return scans_for_fname(ensure_list(val))
        if opt == 'deformation':
            return np.array([simplify_list(val)], dtype=object)
        if opt == 'deformation_field':
            return np.array([simplify_list(val)], dtype=object)
        return val

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_files'] = []
        for filename in self.inputs.in_files:
            _, fname = os.path.split(filename)
            outputs['out_files'].append(os.path.realpath('w%s' % fname))
        return outputs