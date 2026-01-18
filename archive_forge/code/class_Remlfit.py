import os
from ..base import (
from ...external.due import BibTeX
from .base import (
class Remlfit(AFNICommand):
    """Performs Generalized least squares time series fit with Restricted
    Maximum Likelihood (REML) estimation of the temporal auto-correlation
    structure.

    For complete details, see the `3dREMLfit Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dREMLfit.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni
    >>> remlfit = afni.Remlfit()
    >>> remlfit.inputs.in_files = ['functional.nii', 'functional2.nii']
    >>> remlfit.inputs.out_file = 'output.nii'
    >>> remlfit.inputs.matrix = 'output.1D'
    >>> remlfit.inputs.gltsym = [('SYM: +Lab1 -Lab2', 'TestSYM'), ('timeseries.txt', 'TestFile')]
    >>> remlfit.cmdline
    '3dREMLfit -gltsym "SYM: +Lab1 -Lab2" TestSYM -gltsym "timeseries.txt" TestFile -input "functional.nii functional2.nii" -matrix output.1D -Rbuck output.nii'
    >>> res = remlfit.run()  # doctest: +SKIP
    """
    _cmd = '3dREMLfit'
    input_spec = RemlfitInputSpec
    output_spec = RemlfitOutputSpec

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        return super(Remlfit, self)._parse_inputs(skip)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for key in outputs.keys():
            if isdefined(self.inputs.get()[key]):
                outputs[key] = os.path.abspath(self.inputs.get()[key])
        return outputs