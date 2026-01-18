import os
import re as regex
from ..base import (
class Bfc(CommandLine):
    """
    bias field corrector (BFC)
    This program corrects gain variation in T1-weighted MRI.

    http://brainsuite.org/processing/surfaceextraction/bfc/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> bfc = brainsuite.Bfc()
    >>> bfc.inputs.inputMRIFile = example_data('structural.nii')
    >>> bfc.inputs.inputMaskFile = example_data('mask.nii')
    >>> results = bfc.run() #doctest: +SKIP

    """
    input_spec = BfcInputSpec
    output_spec = BfcOutputSpec
    _cmd = 'bfc'

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        fileToSuffixMap = {'outputMRIVolume': '.bfc.nii.gz'}
        if name in fileToSuffixMap:
            return getFileName(self.inputs.inputMRIFile, fileToSuffixMap[name])
        return None

    def _format_arg(self, name, spec, value):
        if name == 'histogramType':
            return spec.argstr % {'ellipse': '--ellipse', 'block': '--block'}[value]
        if name == 'biasRange':
            return spec.argstr % {'low': '--low', 'medium': '--medium', 'high': '--high'}[value]
        if name == 'intermediate_file_type':
            return spec.argstr % {'analyze': '--analyze', 'nifti': '--nifti', 'gzippedAnalyze': '--analyzegz', 'gzippedNifti': '--niftigz'}[value]
        return super(Bfc, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        return l_outputs(self)