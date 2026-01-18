import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class BrainMaskInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='input diffusion weighted images')
    out_file = File('brainmask.mif', argstr='%s', mandatory=True, position=-1, usedefault=True, desc='output brain mask')