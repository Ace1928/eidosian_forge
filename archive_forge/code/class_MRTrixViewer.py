import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class MRTrixViewer(CommandLine):
    """
    Loads the input images in the MRTrix Viewer.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> MRview = mrt.MRTrixViewer()
    >>> MRview.inputs.in_files = 'dwi.mif'
    >>> MRview.run()                                    # doctest: +SKIP
    """
    _cmd = 'mrview'
    input_spec = MRTrixViewerInputSpec
    output_spec = MRTrixViewerOutputSpec

    def _list_outputs(self):
        return