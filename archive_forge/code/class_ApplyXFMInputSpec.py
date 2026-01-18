import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ApplyXFMInputSpec(FLIRTInputSpec):
    apply_xfm = traits.Bool(True, argstr='-applyxfm', desc='apply transformation supplied by in_matrix_file or uses_qform to use the affine matrix stored in the reference header', usedefault=True)