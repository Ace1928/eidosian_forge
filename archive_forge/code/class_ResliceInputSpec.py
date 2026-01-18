import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class ResliceInputSpec(SPMCommandInputSpec):
    in_file = File(exists=True, mandatory=True, desc='file to apply transform to, (only updates header)')
    space_defining = File(exists=True, mandatory=True, desc='Volume defining space to slice in_file into')
    interp = traits.Range(low=0, high=7, usedefault=True, desc='degree of b-spline used for interpolation0 is nearest neighbor (default)')
    out_file = File(desc='Optional file to save resliced volume')