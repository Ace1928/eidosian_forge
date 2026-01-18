import os
from ...base import (
class BRAINSEyeDetector(SEMLikeCommandLine):
    """title: Eye Detector (BRAINS)

    category: Utilities.BRAINS

    version: 1.0

    documentation-url: http://www.nitrc.org/projects/brainscdetector/
    """
    input_spec = BRAINSEyeDetectorInputSpec
    output_spec = BRAINSEyeDetectorOutputSpec
    _cmd = ' BRAINSEyeDetector '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False