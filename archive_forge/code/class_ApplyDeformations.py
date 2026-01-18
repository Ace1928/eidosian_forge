import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class ApplyDeformations(SPMCommand):
    input_spec = ApplyDeformationFieldInputSpec
    output_spec = ApplyDeformationFieldOutputSpec
    _jobtype = 'util'
    _jobname = 'defs'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['deformation_field', 'reference_volume']:
            val = [val]
        if opt in ['deformation_field']:
            return scans_for_fnames(val, keep4d=True, separate_sessions=False)
        if opt in ['in_files', 'reference_volume']:
            return scans_for_fnames(val, keep4d=False, separate_sessions=False)
        else:
            return super(ApplyDeformations, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_files'] = []
        for filename in self.inputs.in_files:
            _, fname = os.path.split(filename)
            outputs['out_files'].append(os.path.realpath('w%s' % fname))
        return outputs