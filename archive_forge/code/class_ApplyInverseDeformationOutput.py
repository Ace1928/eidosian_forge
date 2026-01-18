import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class ApplyInverseDeformationOutput(TraitedSpec):
    out_files = OutputMultiPath(File(exists=True), desc='Transformed files')