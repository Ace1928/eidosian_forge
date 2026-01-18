import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class SliceTimer(FSLCommand):
    """FSL slicetimer wrapper to perform slice timing correction

    Examples
    --------
    >>> from nipype.interfaces import fsl
    >>> from nipype.testing import example_data
    >>> st = fsl.SliceTimer()
    >>> st.inputs.in_file = example_data('functional.nii')
    >>> st.inputs.interleaved = True
    >>> result = st.run() #doctest: +SKIP

    """
    _cmd = 'slicetimer'
    input_spec = SliceTimerInputSpec
    output_spec = SliceTimerOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_file = self.inputs.out_file
        if not isdefined(out_file):
            out_file = self._gen_fname(self.inputs.in_file, suffix='_st')
        outputs['slice_time_corrected_file'] = os.path.abspath(out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()['slice_time_corrected_file']
        return None