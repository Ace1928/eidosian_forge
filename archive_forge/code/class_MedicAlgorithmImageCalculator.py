import os
from ..base import (
class MedicAlgorithmImageCalculator(SEMLikeCommandLine):
    """Perform simple image calculator operations on two images.

    The operations include 'Add', 'Subtract', 'Multiply', and 'Divide'

    """
    input_spec = MedicAlgorithmImageCalculatorInputSpec
    output_spec = MedicAlgorithmImageCalculatorOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run edu.jhu.ece.iacl.plugins.utilities.math.MedicAlgorithmImageCalculator '
    _outputs_filenames = {'outResult': 'outResult.nii'}
    _redirect_x = True