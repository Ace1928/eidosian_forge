import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class SliceTiming(SPMCommand):
    """Use spm to perform slice timing correction.

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=19

    Examples
    --------

    >>> from nipype.interfaces.spm import SliceTiming
    >>> st = SliceTiming()
    >>> st.inputs.in_files = 'functional.nii'
    >>> st.inputs.num_slices = 32
    >>> st.inputs.time_repetition = 6.0
    >>> st.inputs.time_acquisition = 6. - 6./32.
    >>> st.inputs.slice_order = list(range(32,0,-1))
    >>> st.inputs.ref_slice = 1
    >>> st.run() # doctest: +SKIP

    """
    input_spec = SliceTimingInputSpec
    output_spec = SliceTimingOutputSpec
    _jobtype = 'temporal'
    _jobname = 'st'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'in_files':
            return scans_for_fnames(ensure_list(val), keep4d=False, separate_sessions=True)
        return super(SliceTiming, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['timecorrected_files'] = []
        filelist = ensure_list(self.inputs.in_files)
        for f in filelist:
            if isinstance(f, list):
                run = [fname_presuffix(in_f, prefix=self.inputs.out_prefix) for in_f in f]
            else:
                run = fname_presuffix(f, prefix=self.inputs.out_prefix)
            outputs['timecorrected_files'].append(run)
        return outputs