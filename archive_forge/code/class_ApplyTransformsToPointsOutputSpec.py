import os
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, InputMultiObject
from ...utils.filemanip import split_filename
class ApplyTransformsToPointsOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc='csv file with transformed coordinates')