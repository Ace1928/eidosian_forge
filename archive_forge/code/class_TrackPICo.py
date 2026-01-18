import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackPICo(Track):
    """
    Performs streamline tractography using Probabilistic Index of Connectivity (PICo).

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> track = cmon.TrackPICo()
    >>> track.inputs.in_file = 'pdfs.Bfloat'
    >>> track.inputs.seed_file = 'seed_mask.nii'
    >>> track.run()                  # doctest: +SKIP

    """
    input_spec = TrackPICoInputSpec

    def __init__(self, command=None, **inputs):
        inputs['inputmodel'] = 'pico'
        return super(TrackPICo, self).__init__(command, **inputs)