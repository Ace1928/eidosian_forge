import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class DiffusionTensorStreamlineTrack(StreamlineTrack):
    """
    Specialized interface to StreamlineTrack. This interface is used for
    streamline tracking from diffusion tensor data, and calls the MRtrix
    function 'streamtrack' with the option 'DT_STREAM'

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> dtstrack = mrt.DiffusionTensorStreamlineTrack()
    >>> dtstrack.inputs.in_file = 'data.Bfloat'
    >>> dtstrack.inputs.seed_file = 'seed_mask.nii'
    >>> dtstrack.run()                                  # doctest: +SKIP
    """
    input_spec = DiffusionTensorStreamlineTrackInputSpec

    def __init__(self, command=None, **inputs):
        inputs['inputmodel'] = 'DT_STREAM'
        return super(DiffusionTensorStreamlineTrack, self).__init__(command, **inputs)