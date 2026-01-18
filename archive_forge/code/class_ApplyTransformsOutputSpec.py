import os
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, InputMultiObject
from ...utils.filemanip import split_filename
class ApplyTransformsOutputSpec(TraitedSpec):
    output_image = File(exists=True, desc='Warped image')