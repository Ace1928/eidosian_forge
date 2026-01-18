import os
from ...base import (
class FlippedDifference(SEMLikeCommandLine):
    """title: Flip Image

    category: Filtering.FeatureDetection

    description: Difference between an image and the axially flipped version of that image.

    version: 0.1.0.$Revision: 1 $(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was developed by Mark Scully and Jeremy Bockholt.
    """
    input_spec = FlippedDifferenceInputSpec
    output_spec = FlippedDifferenceOutputSpec
    _cmd = ' FlippedDifference '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False