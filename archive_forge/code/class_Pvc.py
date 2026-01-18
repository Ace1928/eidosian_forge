import os
import re as regex
from ..base import (
class Pvc(CommandLine):
    """
    partial volume classifier (PVC) tool.
    This program performs voxel-wise tissue classification T1-weighted MRI.
    Image should be skull-stripped and bias-corrected before tissue classification.

    http://brainsuite.org/processing/surfaceextraction/pvc/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> pvc = brainsuite.Pvc()
    >>> pvc.inputs.inputMRIFile = example_data('structural.nii')
    >>> pvc.inputs.inputMaskFile = example_data('mask.nii')
    >>> results = pvc.run() #doctest: +SKIP

    """
    input_spec = PvcInputSpec
    output_spec = PvcOutputSpec
    _cmd = 'pvc'

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        fileToSuffixMap = {'outputLabelFile': '.pvc.label.nii.gz', 'outputTissueFractionFile': '.pvc.frac.nii.gz'}
        if name in fileToSuffixMap:
            return getFileName(self.inputs.inputMRIFile, fileToSuffixMap[name])
        return None

    def _list_outputs(self):
        return l_outputs(self)