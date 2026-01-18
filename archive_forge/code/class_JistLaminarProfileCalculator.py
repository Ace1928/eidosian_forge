import os
from ..base import (
class JistLaminarProfileCalculator(SEMLikeCommandLine):
    """Compute various moments for intensities mapped along a cortical profile."""
    input_spec = JistLaminarProfileCalculatorInputSpec
    output_spec = JistLaminarProfileCalculatorOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.laminar.JistLaminarProfileCalculator '
    _outputs_filenames = {'outResult': 'outResult.nii'}
    _redirect_x = True