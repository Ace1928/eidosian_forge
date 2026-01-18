import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
def _make_mat_file(self):
    """makes name for matfile if doesn exist"""
    pth, mv, _ = split_filename(self.inputs.moving)
    _, tgt, _ = split_filename(self.inputs.target)
    mat = os.path.join(pth, '%s_to_%s.mat' % (mv, tgt))
    return mat