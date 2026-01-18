import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackDT(Track):
    """
    Performs streamline tractography using tensor data

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> track = cmon.TrackDT()
    >>> track.inputs.in_file = 'tensor_fitted_data.Bdouble'
    >>> track.inputs.seed_file = 'seed_mask.nii'
    >>> track.run()                 # doctest: +SKIP

    """

    def __init__(self, command=None, **inputs):
        inputs['inputmodel'] = 'dt'
        return super(TrackDT, self).__init__(command, **inputs)