import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class ResampleImageBySpacingInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d', desc='dimension of output image')
    input_image = File(exists=True, mandatory=True, position=2, argstr='%s', desc='input image file')
    output_image = File(position=3, argstr='%s', name_source=['input_image'], name_template='%s_resampled', desc='output image file', keep_extension=True)
    out_spacing = traits.Either(traits.List(traits.Float, minlen=2, maxlen=3), traits.Tuple(traits.Float, traits.Float, traits.Float), traits.Tuple(traits.Float, traits.Float), position=4, argstr='%s', mandatory=True, desc='output spacing')
    apply_smoothing = traits.Bool(False, argstr='%d', position=5, desc='smooth before resampling')
    addvox = traits.Int(argstr='%d', position=6, requires=['apply_smoothing'], desc='addvox pads each dimension by addvox')
    nn_interp = traits.Bool(argstr='%d', desc='nn interpolation', position=-1, requires=['addvox'])