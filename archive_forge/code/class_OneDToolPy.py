import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class OneDToolPy(AFNIPythonCommand):
    """This program is meant to read/manipulate/write/diagnose 1D datasets.
    Input can be specified using AFNI sub-brick[]/time{} selectors.

    >>> from nipype.interfaces import afni
    >>> odt = afni.OneDToolPy()
    >>> odt.inputs.in_file = 'f1.1D'
    >>> odt.inputs.set_nruns = 3
    >>> odt.inputs.demean = True
    >>> odt.inputs.out_file = 'motion_dmean.1D'
    >>> odt.cmdline # doctest: +ELLIPSIS
    'python2 ...1d_tool.py -demean -infile f1.1D -write motion_dmean.1D -set_nruns 3'
     >>> res = odt.run()  # doctest: +SKIP"""
    _cmd = '1d_tool.py'
    input_spec = OneDToolPyInputSpec
    output_spec = OneDToolPyOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs['out_file'] = os.path.join(os.getcwd(), self.inputs.out_file)
        if isdefined(self.inputs.show_cormat_warnings):
            outputs['out_file'] = os.path.join(os.getcwd(), self.inputs.show_cormat_warnings)
        if isdefined(self.inputs.censor_motion):
            outputs['out_file'] = os.path.join(os.getcwd(), self.inputs.censor_motion[1] + '_censor.1D')
        return outputs