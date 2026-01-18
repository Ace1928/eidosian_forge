import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Hist(AFNICommandBase):
    """Computes average of all voxels in the input dataset
    which satisfy the criterion in the options list

    For complete details, see the `3dHist Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dHist.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> hist = afni.Hist()
    >>> hist.inputs.in_file = 'functional.nii'
    >>> hist.cmdline
    '3dHist -input functional.nii -prefix functional_hist'
    >>> res = hist.run()  # doctest: +SKIP

    """
    _cmd = '3dHist'
    input_spec = HistInputSpec
    output_spec = HistOutputSpec
    _redirect_x = True

    def __init__(self, **inputs):
        super(Hist, self).__init__(**inputs)
        if not no_afni():
            version = Info.version()
            if version[0] > 2015:
                self._redirect_x = False

    def _parse_inputs(self, skip=None):
        if not self.inputs.showhist:
            if skip is None:
                skip = []
            skip += ['out_show']
        return super(Hist, self)._parse_inputs(skip=skip)

    def _list_outputs(self):
        outputs = super(Hist, self)._list_outputs()
        outputs['out_file'] += '.niml.hist'
        if not self.inputs.showhist:
            outputs['out_show'] = Undefined
        return outputs