import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class SHConv(CommandLine):
    """
    Convolve spherical harmonics with a tissue response function. Useful for
    checking residuals of ODF estimates.


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> sh = mrt.SHConv()
    >>> sh.inputs.in_file = 'csd.mif'
    >>> sh.inputs.response = 'response.txt'
    >>> sh.cmdline
    'shconv csd.mif response.txt csd_shconv.mif'
    >>> sh.run()                                 # doctest: +SKIP
    """
    _cmd = 'shconv'
    input_spec = SHConvInputSpec
    output_spec = SHConvOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs