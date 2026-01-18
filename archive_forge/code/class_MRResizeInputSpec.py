import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class MRResizeInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', position=-2, mandatory=True, desc='input DWI image')
    image_size = traits.Tuple((traits.Int, traits.Int, traits.Int), argstr='-size %d,%d,%d', mandatory=True, desc='Number of voxels in each dimension of output image', xor=['voxel_size', 'scale_factor'])
    voxel_size = traits.Tuple((traits.Float, traits.Float, traits.Float), argstr='-voxel %g,%g,%g', mandatory=True, desc='Desired voxel size in mm for the output image', xor=['image_size', 'scale_factor'])
    scale_factor = traits.Tuple((traits.Float, traits.Float, traits.Float), argstr='-scale %g,%g,%g', mandatory=True, desc='Scale factors to rescale the image by in each dimension', xor=['image_size', 'voxel_size'])
    interpolation = traits.Enum('cubic', 'nearest', 'linear', 'sinc', argstr='-interp %s', usedefault=True, desc='set the interpolation method to use when resizing (choices: nearest, linear, cubic, sinc. Default: cubic).')
    out_file = File(argstr='%s', name_template='%s_resized', name_source=['in_file'], keep_extension=True, position=-1, desc='the output resized DWI image')