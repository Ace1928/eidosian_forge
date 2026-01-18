import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class MNIBiasCorrection(FSCommand):
    """Wrapper for nu_correct, a program from the Montreal Neurological Insitute (MNI)
    used for correcting intensity non-uniformity (ie, bias fields). You must have the
    MNI software installed on your system to run this. See [www.bic.mni.mcgill.ca/software/N3]
    for more info.

    mri_nu_correct.mni uses float internally instead of uchar. It also rescales the output so
    that the global mean is the same as that of the input. These two changes are linked and
    can be turned off with --no-float

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import MNIBiasCorrection
    >>> correct = MNIBiasCorrection()
    >>> correct.inputs.in_file = "norm.mgz"
    >>> correct.inputs.iterations = 6
    >>> correct.inputs.protocol_iterations = 1000
    >>> correct.inputs.distance = 50
    >>> correct.cmdline
    'mri_nu_correct.mni --distance 50 --i norm.mgz --n 6 --o norm_output.mgz --proto-iters 1000'

    References
    ----------
    [http://freesurfer.net/fswiki/mri_nu_correct.mni]
    [http://www.bic.mni.mcgill.ca/software/N3]
    [https://github.com/BIC-MNI/N3]

    """
    _cmd = 'mri_nu_correct.mni'
    input_spec = MNIBiasCorrectionInputSpec
    output_spec = MNIBiasCorrectionOutputSpec