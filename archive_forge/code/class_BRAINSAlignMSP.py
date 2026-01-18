import os
from ...base import (
class BRAINSAlignMSP(SEMLikeCommandLine):
    """title: Align Mid Sagittal Brain (BRAINS)

    category: Utilities.BRAINS

    description: Resample an image into ACPC alignment ACPCDetect
    """
    input_spec = BRAINSAlignMSPInputSpec
    output_spec = BRAINSAlignMSPOutputSpec
    _cmd = ' BRAINSAlignMSP '
    _outputs_filenames = {'OutputresampleMSP': 'OutputresampleMSP.nii', 'resultsDir': 'resultsDir'}
    _redirect_x = False