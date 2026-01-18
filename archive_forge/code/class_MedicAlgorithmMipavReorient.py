import os
from ..base import (
class MedicAlgorithmMipavReorient(SEMLikeCommandLine):
    """Reorient a volume to a particular anatomical orientation."""
    input_spec = MedicAlgorithmMipavReorientInputSpec
    output_spec = MedicAlgorithmMipavReorientOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run edu.jhu.ece.iacl.plugins.utilities.volume.MedicAlgorithmMipavReorient '
    _outputs_filenames = {}
    _redirect_x = True