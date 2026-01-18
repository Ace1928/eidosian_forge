import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class MRCat(CommandLine):
    """
    Concatenate several images into one


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> mrcat = mrt.MRCat()
    >>> mrcat.inputs.in_files = ['dwi.mif','mask.mif']
    >>> mrcat.cmdline                               # doctest: +ELLIPSIS
    'mrcat dwi.mif mask.mif concatenated.mif'
    >>> mrcat.run()                                 # doctest: +SKIP
    """
    _cmd = 'mrcat'
    input_spec = MRCatInputSpec
    output_spec = MRCatOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs