import os
from ...base import (
class GenerateBrainClippedImage(SEMLikeCommandLine):
    """title: GenerateBrainClippedImage

    category: Filtering.FeatureDetection

    description: Automatic FeatureImages using neural networks

    version: 1.0

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Eun Young Kim
    """
    input_spec = GenerateBrainClippedImageInputSpec
    output_spec = GenerateBrainClippedImageOutputSpec
    _cmd = ' GenerateBrainClippedImage '
    _outputs_filenames = {'outputFileName': 'outputFileName'}
    _redirect_x = False