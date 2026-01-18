import os
import re as regex
from ..base import (
class Scrubmask(CommandLine):
    """
    ScrubMask tool
    scrubmask filters binary masks to trim loosely connected voxels that may
    result from segmentation errors and produce bumps on tessellated surfaces.

    http://brainsuite.org/processing/surfaceextraction/scrubmask/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> scrubmask = brainsuite.Scrubmask()
    >>> scrubmask.inputs.inputMaskFile = example_data('mask.nii')
    >>> results = scrubmask.run() #doctest: +SKIP

    """
    input_spec = ScrubmaskInputSpec
    output_spec = ScrubmaskOutputSpec
    _cmd = 'scrubmask'

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        if name == 'outputMaskFile':
            return getFileName(self.inputs.inputMaskFile, '.cortex.scrubbed.mask.nii.gz')
        return None

    def _list_outputs(self):
        return l_outputs(self)