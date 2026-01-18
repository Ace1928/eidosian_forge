import os
import re as regex
from ..base import (
class Skullfinder(CommandLine):
    """
    Skull and scalp segmentation algorithm.

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> skullfinder = brainsuite.Skullfinder()
    >>> skullfinder.inputs.inputMRIFile = example_data('structural.nii')
    >>> skullfinder.inputs.inputMaskFile = example_data('mask.nii')
    >>> results = skullfinder.run() #doctest: +SKIP

    """
    input_spec = SkullfinderInputSpec
    output_spec = SkullfinderOutputSpec
    _cmd = 'skullfinder'

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        if name == 'outputLabelFile':
            return getFileName(self.inputs.inputMRIFile, '.skullfinder.label.nii.gz')
        return None

    def _list_outputs(self):
        return l_outputs(self)