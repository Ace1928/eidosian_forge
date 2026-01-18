import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class SmoothEstimate(FSLCommand):
    """Estimates the smoothness of an image

    Examples
    --------

    >>> est = SmoothEstimate()
    >>> est.inputs.zstat_file = 'zstat1.nii.gz'
    >>> est.inputs.mask_file = 'mask.nii'
    >>> est.cmdline
    'smoothest --mask=mask.nii --zstat=zstat1.nii.gz'

    """
    input_spec = SmoothEstimateInputSpec
    output_spec = SmoothEstimateOutputSpec
    _cmd = 'smoothest'

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = self._outputs()
        stdout = runtime.stdout.split('\n')
        outputs.dlh = float(stdout[0].split()[1])
        outputs.volume = int(stdout[1].split()[1])
        outputs.resels = float(stdout[2].split()[1])
        return outputs