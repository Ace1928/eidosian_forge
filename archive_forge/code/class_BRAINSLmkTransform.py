import os
from ...base import (
class BRAINSLmkTransform(SEMLikeCommandLine):
    """title: Landmark Transform (BRAINS)

    category: Utilities.BRAINS

    description: This utility program estimates the affine transform to align the fixed landmarks to the moving landmarks, and then generate the resampled moving image to the same physical space as that of the reference image.

    version: 1.0

    documentation-url: http://www.nitrc.org/projects/brainscdetector/
    """
    input_spec = BRAINSLmkTransformInputSpec
    output_spec = BRAINSLmkTransformOutputSpec
    _cmd = ' BRAINSLmkTransform '
    _outputs_filenames = {'outputResampledVolume': 'outputResampledVolume.nii', 'outputAffineTransform': 'outputAffineTransform.h5'}
    _redirect_x = False