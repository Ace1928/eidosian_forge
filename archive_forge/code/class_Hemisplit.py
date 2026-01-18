import os
import re as regex
from ..base import (
class Hemisplit(CommandLine):
    """
    Hemisphere splitter
    Splits a surface object into two separate surfaces given an input label volume.
    Each vertex is labeled left or right based on the labels being odd (left) or even (right).
    The largest contour on the split surface is then found and used as the separation between left and right.

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> hemisplit = brainsuite.Hemisplit()
    >>> hemisplit.inputs.inputSurfaceFile = 'input_surf.dfs'
    >>> hemisplit.inputs.inputHemisphereLabelFile = 'label.nii'
    >>> hemisplit.inputs.pialSurfaceFile = 'pial.dfs'
    >>> results = hemisplit.run() #doctest: +SKIP

    """
    input_spec = HemisplitInputSpec
    output_spec = HemisplitOutputSpec
    _cmd = 'hemisplit'

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        fileToSuffixMap = {'outputLeftHemisphere': '.left.inner.cortex.dfs', 'outputLeftPialHemisphere': '.left.pial.cortex.dfs', 'outputRightHemisphere': '.right.inner.cortex.dfs', 'outputRightPialHemisphere': '.right.pial.cortex.dfs'}
        if name in fileToSuffixMap:
            return getFileName(self.inputs.inputSurfaceFile, fileToSuffixMap[name])
        return None

    def _list_outputs(self):
        return l_outputs(self)