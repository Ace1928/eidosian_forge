import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class AverageAffineTransformOutputSpec(TraitedSpec):
    affine_transform = File(exists=True, desc='average transform file')