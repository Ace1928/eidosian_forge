import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class ProbabilisticSphericallyDeconvolutedStreamlineTrack(StreamlineTrack):
    """
    Performs probabilistic tracking using spherically deconvolved data

    Specialized interface to StreamlineTrack. This interface is used for
    probabilistic tracking from spherically deconvolved data, and calls
    the MRtrix function 'streamtrack' with the option 'SD_PROB'

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> sdprobtrack = mrt.ProbabilisticSphericallyDeconvolutedStreamlineTrack()
    >>> sdprobtrack.inputs.in_file = 'data.Bfloat'
    >>> sdprobtrack.inputs.seed_file = 'seed_mask.nii'
    >>> sdprobtrack.run()                                                       # doctest: +SKIP
    """
    input_spec = ProbabilisticSphericallyDeconvolutedStreamlineTrackInputSpec

    def __init__(self, command=None, **inputs):
        inputs['inputmodel'] = 'SD_PROB'
        return super(ProbabilisticSphericallyDeconvolutedStreamlineTrack, self).__init__(command, **inputs)