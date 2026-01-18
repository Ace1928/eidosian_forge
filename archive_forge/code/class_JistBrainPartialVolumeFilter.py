import os
from ..base import (
class JistBrainPartialVolumeFilter(SEMLikeCommandLine):
    """Partial Volume Filter.

    Filters an image for regions of partial voluming assuming a ridge-like model of intensity.

    """
    input_spec = JistBrainPartialVolumeFilterInputSpec
    output_spec = JistBrainPartialVolumeFilterOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.brain.JistBrainPartialVolumeFilter '
    _outputs_filenames = {'outPartial': 'outPartial.nii'}
    _redirect_x = True