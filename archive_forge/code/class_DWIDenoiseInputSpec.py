import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class DWIDenoiseInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', position=-2, mandatory=True, desc='input DWI image')
    mask = File(exists=True, argstr='-mask %s', position=1, desc='mask image')
    extent = traits.Tuple((traits.Int, traits.Int, traits.Int), argstr='-extent %d,%d,%d', desc='set the window size of the denoising filter. (default = 5,5,5)')
    noise = File(argstr='-noise %s', name_template='%s_noise', name_source='in_file', keep_extension=True, desc='the output noise map')
    out_file = File(argstr='%s', position=-1, name_template='%s_denoised', name_source='in_file', keep_extension=True, desc='the output denoised DWI image')