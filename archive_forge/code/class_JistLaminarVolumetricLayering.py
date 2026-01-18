import os
from ..base import (
class JistLaminarVolumetricLayering(SEMLikeCommandLine):
    """Volumetric Layering.

    Builds a continuous layering of the cortex following distance-preserving or volume-preserving
    models of cortical folding.

    References
    ----------
    Waehnert MD, Dinse J, Weiss M, Streicher MN, Waehnert P, Geyer S, Turner R, Bazin PL,
    Anatomically motivated modeling of cortical laminae, Neuroimage, 2013.

    """
    input_spec = JistLaminarVolumetricLayeringInputSpec
    output_spec = JistLaminarVolumetricLayeringOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.laminar.JistLaminarVolumetricLayering '
    _outputs_filenames = {'outContinuous': 'outContinuous.nii', 'outLayer': 'outLayer.nii', 'outDiscrete': 'outDiscrete.nii'}
    _redirect_x = True