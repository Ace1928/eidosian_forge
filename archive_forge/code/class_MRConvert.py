import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class MRConvert(MRTrix3Base):
    """
    Perform conversion between different file types and optionally extract a
    subset of the input image

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mrconvert = mrt.MRConvert()
    >>> mrconvert.inputs.in_file = 'dwi.nii.gz'
    >>> mrconvert.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> mrconvert.cmdline                             # doctest: +ELLIPSIS
    'mrconvert -fslgrad bvecs bvals dwi.nii.gz dwi.mif'
    >>> mrconvert.run()                               # doctest: +SKIP
    """
    _cmd = 'mrconvert'
    input_spec = MRConvertInputSpec
    output_spec = MRConvertOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        if self.inputs.json_export:
            outputs['json_export'] = op.abspath(self.inputs.json_export)
        if self.inputs.out_bvec:
            outputs['out_bvec'] = op.abspath(self.inputs.out_bvec)
        if self.inputs.out_bval:
            outputs['out_bval'] = op.abspath(self.inputs.out_bval)
        return outputs