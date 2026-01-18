import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class ResliceToReference(SPMCommand):
    """Uses spm to reslice a volume to a target image space or to a provided
    voxel size and bounding box

    Examples
    --------

    >>> import nipype.interfaces.spm.utils as spmu
    >>> r2ref = spmu.ResliceToReference()
    >>> r2ref.inputs.in_files = 'functional.nii'
    >>> r2ref.inputs.target = 'structural.nii'
    >>> r2ref.run() # doctest: +SKIP
    """
    input_spec = ResliceToReferenceInput
    output_spec = ResliceToReferenceOutput
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