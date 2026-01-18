import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ConvertWarp(FSLCommand):
    """Use FSL `convertwarp <http://fsl.fmrib.ox.ac.uk/fsl/fsl-4.1.9/fnirt/warp_utils.html>`_
    for combining multiple transforms into one.


    Examples
    --------

    >>> from nipype.interfaces.fsl import ConvertWarp
    >>> warputils = ConvertWarp()
    >>> warputils.inputs.warp1 = "warpfield.nii"
    >>> warputils.inputs.reference = "T1.nii"
    >>> warputils.inputs.relwarp = True
    >>> warputils.inputs.output_type = "NIFTI_GZ"
    >>> warputils.cmdline # doctest: +ELLIPSIS
    'convertwarp --ref=T1.nii --rel --warp1=warpfield.nii --out=T1_concatwarp.nii.gz'
    >>> res = warputils.run() # doctest: +SKIP


    """
    input_spec = ConvertWarpInputSpec
    output_spec = ConvertWarpOutputSpec
    _cmd = 'convertwarp'