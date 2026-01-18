import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class DWI2Tensor(CommandLine):
    """
    Converts diffusion-weighted images to tensor images.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> dwi2tensor = mrt.DWI2Tensor()
    >>> dwi2tensor.inputs.in_file = 'dwi.mif'
    >>> dwi2tensor.inputs.encoding_file = 'encoding.txt'
    >>> dwi2tensor.cmdline
    'dwi2tensor -grad encoding.txt dwi.mif dwi_tensor.mif'
    >>> dwi2tensor.run()                                   # doctest: +SKIP
    """
    _cmd = 'dwi2tensor'
    input_spec = DWI2TensorInputSpec
    output_spec = DWI2TensorOutputSpec