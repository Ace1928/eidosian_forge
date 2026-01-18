import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
def _use_cuda(self):
    self._cmd = 'eddy_cuda' if self.inputs.use_cuda else 'eddy_openmp'