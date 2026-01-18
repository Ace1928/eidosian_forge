import os
import os.path as op
from ..base import CommandLineInputSpec, traits, TraitedSpec, File, isdefined
from .base import MRTrix3Base
class LabelConvertOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')