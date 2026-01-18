import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class AverageAffineTransformInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='%d', mandatory=True, position=0, desc='image dimension (2 or 3)')
    output_affine_transform = File(argstr='%s', mandatory=True, position=1, desc='Outputfname.txt: the name of the resulting transform.')
    transforms = InputMultiObject(File(exists=True), argstr='%s', mandatory=True, position=3, desc='transforms to average')