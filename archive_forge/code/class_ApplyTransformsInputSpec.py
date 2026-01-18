import os
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, InputMultiObject
from ...utils.filemanip import split_filename
class ApplyTransformsInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(2, 3, 4, argstr='--dimensionality %d', desc='This option forces the image to be treated as a specified-dimensional image. If not specified, antsWarp tries to infer the dimensionality from the input image.')
    input_image_type = traits.Enum(0, 1, 2, 3, argstr='--input-image-type %d', desc='Option specifying the input image type of scalar (default), vector, tensor, or time series.')
    input_image = File(argstr='--input %s', mandatory=True, desc='image to apply transformation to (generally a coregistered functional)', exists=True)
    output_image = traits.Str(argstr='--output %s', desc='output file name', genfile=True, hash_files=False)
    out_postfix = traits.Str('_trans', usedefault=True, desc='Postfix that is appended to all output files (default = _trans)')
    reference_image = File(argstr='--reference-image %s', mandatory=True, desc='reference image space that you wish to warp INTO', exists=True)
    interpolation = traits.Enum('Linear', 'NearestNeighbor', 'CosineWindowedSinc', 'WelchWindowedSinc', 'HammingWindowedSinc', 'LanczosWindowedSinc', 'MultiLabel', 'Gaussian', 'BSpline', argstr='%s', usedefault=True)
    interpolation_parameters = traits.Either(traits.Tuple(traits.Int()), traits.Tuple(traits.Float(), traits.Float()))
    transforms = InputMultiObject(traits.Either(File(exists=True), 'identity'), argstr='%s', mandatory=True, desc='transform files: will be applied in reverse order. For example, the last specified transform will be applied first.')
    invert_transform_flags = InputMultiObject(traits.Bool())
    default_value = traits.Float(0.0, argstr='--default-value %g', usedefault=True)
    print_out_composite_warp_file = traits.Bool(False, requires=['output_image'], desc='output a composite warp file instead of a transformed image')
    float = traits.Bool(argstr='--float %d', default_value=False, usedefault=True, desc='Use float instead of double for computations.')