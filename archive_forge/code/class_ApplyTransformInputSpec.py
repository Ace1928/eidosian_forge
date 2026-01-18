import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class ApplyTransformInputSpec(SPMCommandInputSpec):
    in_file = File(exists=True, mandatory=True, copyfile=True, desc='file to apply transform to, (only updates header)')
    mat = File(exists=True, mandatory=True, desc='file holding transform to apply')
    out_file = File(desc='output file name for transformed data', genfile=True)