import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class ACTPrepareFSLOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')