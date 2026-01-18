import os
from ...base import (
class ESLR(SEMLikeCommandLine):
    """title: Clean Contiguous Label Map (BRAINS)

    category: Segmentation.Specialized

    description: From a range of label map values, extract the largest contiguous region of those labels
    """
    input_spec = ESLRInputSpec
    output_spec = ESLROutputSpec
    _cmd = ' ESLR '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii.gz'}
    _redirect_x = False