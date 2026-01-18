import os
from ...base import (
class NeighborhoodMedian(SEMLikeCommandLine):
    """title: Neighborhood Median

    category: Filtering.FeatureDetection

    description: Calculates the median, for the given neighborhood size, at each voxel of the input image.

    version: 0.1.0.$Revision: 1 $(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was developed by Mark Scully and Jeremy Bockholt.
    """
    input_spec = NeighborhoodMedianInputSpec
    output_spec = NeighborhoodMedianOutputSpec
    _cmd = ' NeighborhoodMedian '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False