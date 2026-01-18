import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class CalcCoregAffineInputSpec(SPMCommandInputSpec):
    target = File(exists=True, mandatory=True, desc='target for generating affine transform')
    moving = File(exists=True, mandatory=True, copyfile=False, desc='volume transform can be applied to register with target')
    mat = File(desc='Filename used to store affine matrix')
    invmat = File(desc='Filename used to store inverse affine matrix')