import os
from ..base import (
class JistBrainMp2rageSkullStripping(SEMLikeCommandLine):
    """Estimate a brain mask for a MP2RAGE dataset.

    At least a T1-weighted or a T1 map image is required.

    """
    input_spec = JistBrainMp2rageSkullStrippingInputSpec
    output_spec = JistBrainMp2rageSkullStrippingOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.brain.JistBrainMp2rageSkullStripping '
    _outputs_filenames = {'outBrain': 'outBrain.nii', 'outMasked3': 'outMasked3.nii', 'outMasked2': 'outMasked2.nii', 'outMasked': 'outMasked.nii'}
    _redirect_x = True