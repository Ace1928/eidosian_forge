import os
from ..base import (
class MedicAlgorithmThresholdToBinaryMask(SEMLikeCommandLine):
    """Threshold to Binary Mask.

    Given a volume and an intensity range create a binary mask for values within that range.

    """
    input_spec = MedicAlgorithmThresholdToBinaryMaskInputSpec
    output_spec = MedicAlgorithmThresholdToBinaryMaskOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run edu.jhu.ece.iacl.plugins.utilities.volume.MedicAlgorithmThresholdToBinaryMask '
    _outputs_filenames = {}
    _redirect_x = True