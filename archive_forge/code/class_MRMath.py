import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class MRMath(MRTrix3Base):
    """
    Compute summary statistic on image intensities
    along a specified axis of a single image

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mrmath = mrt.MRMath()
    >>> mrmath.inputs.in_file = 'dwi.mif'
    >>> mrmath.inputs.operation = 'mean'
    >>> mrmath.inputs.axis = 3
    >>> mrmath.inputs.out_file = 'dwi_mean.mif'
    >>> mrmath.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> mrmath.cmdline                             # doctest: +ELLIPSIS
    'mrmath -axis 3 -fslgrad bvecs bvals dwi.mif mean dwi_mean.mif'
    >>> mrmath.run()                               # doctest: +SKIP
    """
    _cmd = 'mrmath'
    input_spec = MRMathInputSpec
    output_spec = MRMathOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs