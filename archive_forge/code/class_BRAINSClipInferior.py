import os
from ...base import (
class BRAINSClipInferior(SEMLikeCommandLine):
    """title: Clip Inferior of Center of Brain (BRAINS)

    category: Utilities.BRAINS

    description: This program will read the inputVolume as a short int image, write the BackgroundFillValue everywhere inferior to the lower bound, and write the resulting clipped short int image in the outputVolume.

    version: 1.0
    """
    input_spec = BRAINSClipInferiorInputSpec
    output_spec = BRAINSClipInferiorOutputSpec
    _cmd = ' BRAINSClipInferior '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False