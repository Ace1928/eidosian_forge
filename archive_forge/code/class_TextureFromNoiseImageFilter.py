import os
from ...base import (
class TextureFromNoiseImageFilter(SEMLikeCommandLine):
    """title: TextureFromNoiseImageFilter

    category: Filtering.FeatureDetection

    description: Calculate the local noise in an image.

    version: 0.1.0.$Revision: 1 $(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was developed by Eunyoung Regina Kim
    """
    input_spec = TextureFromNoiseImageFilterInputSpec
    output_spec = TextureFromNoiseImageFilterOutputSpec
    _cmd = ' TextureFromNoiseImageFilter '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False