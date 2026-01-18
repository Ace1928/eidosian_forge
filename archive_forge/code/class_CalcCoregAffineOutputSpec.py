import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class CalcCoregAffineOutputSpec(TraitedSpec):
    mat = File(exists=True, desc='Matlab file holding transform')
    invmat = File(desc='Matlab file holding inverse transform')