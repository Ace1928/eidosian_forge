import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class Directions2Amplitude(CommandLine):
    """
    convert directions image to amplitudes

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> amplitudes = mrt.Directions2Amplitude()
    >>> amplitudes.inputs.in_file = 'peak_directions.mif'
    >>> amplitudes.run()                                          # doctest: +SKIP
    """
    _cmd = 'dir2amp'
    input_spec = Directions2AmplitudeInputSpec
    output_spec = Directions2AmplitudeOutputSpec