import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class InvWarp(FSLCommand):
    """
    Use FSL Invwarp to invert a FNIRT warp


    Examples
    --------

    >>> from nipype.interfaces.fsl import InvWarp
    >>> invwarp = InvWarp()
    >>> invwarp.inputs.warp = "struct2mni.nii"
    >>> invwarp.inputs.reference = "anatomical.nii"
    >>> invwarp.inputs.output_type = "NIFTI_GZ"
    >>> invwarp.cmdline
    'invwarp --out=struct2mni_inverse.nii.gz --ref=anatomical.nii --warp=struct2mni.nii'
    >>> res = invwarp.run() # doctest: +SKIP


    """
    input_spec = InvWarpInputSpec
    output_spec = InvWarpOutputSpec
    _cmd = 'invwarp'