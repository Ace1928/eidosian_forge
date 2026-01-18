import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class FindShPeaks(CommandLine):
    """
    identify the orientations of the N largest peaks of a SH profile

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> shpeaks = mrt.FindShPeaks()
    >>> shpeaks.inputs.in_file = 'csd.mif'
    >>> shpeaks.inputs.directions_file = 'dirs.txt'
    >>> shpeaks.inputs.num_peaks = 2
    >>> shpeaks.run()                                          # doctest: +SKIP
    """
    _cmd = 'find_SH_peaks'
    input_spec = FindShPeaksInputSpec
    output_spec = FindShPeaksOutputSpec