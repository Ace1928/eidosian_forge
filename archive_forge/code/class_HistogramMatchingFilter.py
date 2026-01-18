import os
from ...base import (
class HistogramMatchingFilter(SEMLikeCommandLine):
    """title: Write Out Image Intensities

    category: BRAINS.Utilities

    description: For Analysis

    version: 0.1

    contributor: University of Iowa Department of Psychiatry, http:://www.psychiatry.uiowa.edu
    """
    input_spec = HistogramMatchingFilterInputSpec
    output_spec = HistogramMatchingFilterOutputSpec
    _cmd = ' HistogramMatchingFilter '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False