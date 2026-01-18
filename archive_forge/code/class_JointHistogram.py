import os
from ...base import (
class JointHistogram(SEMLikeCommandLine):
    """title: Write Out Image Intensities

    category: Utilities.BRAINS

    description: For Analysis

    version: 0.1

    contributor: University of Iowa Department of Psychiatry, http:://www.psychiatry.uiowa.edu
    """
    input_spec = JointHistogramInputSpec
    output_spec = JointHistogramOutputSpec
    _cmd = ' JointHistogram '
    _outputs_filenames = {}
    _redirect_x = False