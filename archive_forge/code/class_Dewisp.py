import os
import re as regex
from ..base import (
class Dewisp(CommandLine):
    """
    dewisp
    removes wispy tendril structures from cortex model binary masks.
    It does so based on graph theoretic analysis of connected components,
    similar to TCA. Each branch of the structure graph is analyzed to determine
    pinch points that indicate a likely error in segmentation that attaches noise
    to the image. The pinch threshold determines how many voxels the cross-section
    can be before it is considered part of the image.

    http://brainsuite.org/processing/surfaceextraction/dewisp/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> dewisp = brainsuite.Dewisp()
    >>> dewisp.inputs.inputMaskFile = example_data('mask.nii')
    >>> results = dewisp.run() #doctest: +SKIP

    """
    input_spec = DewispInputSpec
    output_spec = DewispOutputSpec
    _cmd = 'dewisp'

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        if name == 'outputMaskFile':
            return getFileName(self.inputs.inputMaskFile, '.cortex.dewisp.mask.nii.gz')
        return None

    def _list_outputs(self):
        return l_outputs(self)