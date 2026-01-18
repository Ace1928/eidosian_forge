import os
from ... import logging
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .model import GLMFitInputSpec, GLMFit
class MRTM(GLMFit):
    """Perform MRTM1 kinetic modeling.

    Examples
    --------
    >>> mrtm = MRTM()
    >>> mrtm.inputs.in_file = 'tac.nii'
    >>> mrtm.inputs.mrtm1 = ('ref_tac.dat', 'timing.dat')
    >>> mrtm.inputs.glm_dir = 'mrtm'
    >>> mrtm.cmdline
    'mri_glmfit --glmdir mrtm --y tac.nii --mrtm1 ref_tac.dat timing.dat'
    """
    input_spec = MRTMInputSpec