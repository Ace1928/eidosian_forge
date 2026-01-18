import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackBootstrap(Track):
    """
    Performs bootstrap streamline tractography using multiple scans of the same subject

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> track = cmon.TrackBootstrap()
    >>> track.inputs.inputmodel='repbs_dt'
    >>> track.inputs.scheme_file = 'bvecs.scheme'
    >>> track.inputs.bsdatafiles = ['fitted_data1.Bfloat', 'fitted_data2.Bfloat']
    >>> track.inputs.seed_file = 'seed_mask.nii'
    >>> track.run()                  # doctest: +SKIP

    """
    input_spec = TrackBootstrapInputSpec

    def __init__(self, command=None, **inputs):
        return super(TrackBootstrap, self).__init__(command, **inputs)