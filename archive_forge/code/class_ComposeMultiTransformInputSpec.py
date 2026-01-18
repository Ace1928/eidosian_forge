import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class ComposeMultiTransformInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', usedefault=True, position=0, desc='image dimension (2 or 3)')
    output_transform = File(argstr='%s', position=1, name_source=['transforms'], name_template='%s_composed', keep_extension=True, desc='the name of the resulting transform.')
    reference_image = File(argstr='%s', position=2, desc='Reference image (only necessary when output is warpfield)')
    transforms = InputMultiObject(File(exists=True), argstr='%s', mandatory=True, position=3, desc='transforms to average')