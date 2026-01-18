import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegResampleInputSpec(NiftyRegCommandInputSpec):
    """Input Spec for RegResample."""
    ref_file = File(exists=True, desc='The input reference/target image', argstr='-ref %s', mandatory=True)
    flo_file = File(exists=True, desc='The input floating/source image', argstr='-flo %s', mandatory=True)
    trans_file = File(exists=True, desc='The input transformation file', argstr='-trans %s')
    type = traits.Enum('res', 'blank', argstr='-%s', position=-2, usedefault=True, desc='Type of output')
    out_file = File(name_source=['flo_file'], name_template='%s', argstr='%s', position=-1, desc='The output filename of the transformed image')
    inter_val = traits.Enum('NN', 'LIN', 'CUB', 'SINC', desc='Interpolation type', argstr='-inter %d')
    pad_val = traits.Float(desc='Padding value', argstr='-pad %f')
    tensor_flag = traits.Bool(desc='Resample Tensor Map', argstr='-tensor ')
    verbosity_off_flag = traits.Bool(argstr='-voff', desc='Turn off verbose output')
    desc = 'Perform the resampling in two steps to resample an image to a lower resolution'
    psf_flag = traits.Bool(argstr='-psf', desc=desc)
    desc = 'Minimise the matrix metric (0) or the determinant (1) when estimating the PSF [0]'
    psf_alg = traits.Enum(0, 1, argstr='-psf_alg %d', desc=desc)