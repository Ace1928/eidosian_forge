import os
from ..base import (
class JistBrainMgdmSegmentation(SEMLikeCommandLine):
    """MGDM Whole Brain Segmentation.

    Estimate brain structures from an atlas for a MRI dataset (multiple input combinations
    are possible).

    """
    input_spec = JistBrainMgdmSegmentationInputSpec
    output_spec = JistBrainMgdmSegmentationOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.brain.JistBrainMgdmSegmentation '
    _outputs_filenames = {'outSegmented': 'outSegmented.nii', 'outPosterior2': 'outPosterior2.nii', 'outPosterior3': 'outPosterior3.nii', 'outLevelset': 'outLevelset.nii'}
    _redirect_x = True