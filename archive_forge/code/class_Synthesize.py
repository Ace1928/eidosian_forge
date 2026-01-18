import os
from ..base import (
from ...external.due import BibTeX
from .base import (
class Synthesize(AFNICommand):
    """Reads a '-cbucket' dataset and a '.xmat.1D' matrix from 3dDeconvolve,
       and synthesizes a fit dataset using user-selected sub-bricks and
       matrix columns.

    For complete details, see the `3dSynthesize Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dSynthesize.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni
    >>> synthesize = afni.Synthesize()
    >>> synthesize.inputs.cbucket = 'functional.nii'
    >>> synthesize.inputs.matrix = 'output.1D'
    >>> synthesize.inputs.select = ['baseline']
    >>> synthesize.cmdline
    '3dSynthesize -cbucket functional.nii -matrix output.1D -select baseline'
    >>> syn = synthesize.run()  # doctest: +SKIP
    """
    _cmd = '3dSynthesize'
    input_spec = SynthesizeInputSpec
    output_spec = AFNICommandOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for key in outputs.keys():
            if isdefined(self.inputs.get()[key]):
                outputs[key] = os.path.abspath(self.inputs.get()[key])
        return outputs