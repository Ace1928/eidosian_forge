import os
import re as regex
from ..base import (
class BDP(CommandLine):
    """
    BrainSuite Diffusion Pipeline (BDP) enables fusion of diffusion and
    structural MRI information for advanced image and connectivity analysis.
    It provides various methods for distortion correction, co-registration,
    diffusion modeling (DTI and ODF) and basic ROI-wise statistic. BDP is a
    flexible and diverse tool which supports wide variety of diffusion
    datasets.
    For more information, please see:

    http://brainsuite.org/processing/diffusion/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> bdp = brainsuite.BDP()
    >>> bdp.inputs.bfcFile = '/directory/subdir/prefix.bfc.nii.gz'
    >>> bdp.inputs.inputDiffusionData = '/directory/subdir/prefix.dwi.nii.gz'
    >>> bdp.inputs.BVecBValPair = ['/directory/subdir/prefix.dwi.bvec', '/directory/subdir/prefix.dwi.bval']
    >>> results = bdp.run() #doctest: +SKIP


    """
    input_spec = BDPInputSpec
    _cmd = 'bdp.sh'

    def _format_arg(self, name, spec, value):
        if name == 'BVecBValPair':
            return spec.argstr % (value[0], value[1])
        if name == 'dataSinkDelay':
            return spec.argstr % ''
        return super(BDP, self)._format_arg(name, spec, value)