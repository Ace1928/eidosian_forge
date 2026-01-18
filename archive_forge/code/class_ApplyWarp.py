import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ApplyWarp(FSLCommand):
    """FSL's applywarp wrapper to apply the results of a FNIRT registration

    Examples
    --------
    >>> from nipype.interfaces import fsl
    >>> from nipype.testing import example_data
    >>> aw = fsl.ApplyWarp()
    >>> aw.inputs.in_file = example_data('structural.nii')
    >>> aw.inputs.ref_file = example_data('mni.nii')
    >>> aw.inputs.field_file = 'my_coefficients_filed.nii' #doctest: +SKIP
    >>> res = aw.run() #doctest: +SKIP


    """
    _cmd = 'applywarp'
    input_spec = ApplyWarpInputSpec
    output_spec = ApplyWarpOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'superlevel':
            return spec.argstr % str(value)
        return super(ApplyWarp, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        if not isdefined(self.inputs.out_file):
            outputs['out_file'] = self._gen_fname(self.inputs.in_file, suffix='_warp')
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None