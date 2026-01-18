import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class MRTrixInfo(CommandLine):
    """
    Prints out relevant header information found in the image specified.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> MRinfo = mrt.MRTrixInfo()
    >>> MRinfo.inputs.in_file = 'dwi.mif'
    >>> MRinfo.run()                                    # doctest: +SKIP
    """
    _cmd = 'mrinfo'
    input_spec = MRTrixInfoInputSpec
    output_spec = MRTrixInfoOutputSpec

    def _list_outputs(self):
        return