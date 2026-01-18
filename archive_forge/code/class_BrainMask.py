import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class BrainMask(CommandLine):
    """
    Convert a mesh surface to a partial volume estimation image


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> bmsk = mrt.BrainMask()
    >>> bmsk.inputs.in_file = 'dwi.mif'
    >>> bmsk.cmdline                               # doctest: +ELLIPSIS
    'dwi2mask dwi.mif brainmask.mif'
    >>> bmsk.run()                                 # doctest: +SKIP
    """
    _cmd = 'dwi2mask'
    input_spec = BrainMaskInputSpec
    output_spec = BrainMaskOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs