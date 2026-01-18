import os
from ...base import (
class BRAINSTrimForegroundInDirection(SEMLikeCommandLine):
    """title: Trim Foreground In Direction (BRAINS)

    category: Utilities.BRAINS

    description: This program will trim off the neck and also air-filling noise from the inputImage.

    version: 0.1

    documentation-url: http://www.nitrc.org/projects/art/
    """
    input_spec = BRAINSTrimForegroundInDirectionInputSpec
    output_spec = BRAINSTrimForegroundInDirectionOutputSpec
    _cmd = ' BRAINSTrimForegroundInDirection '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False