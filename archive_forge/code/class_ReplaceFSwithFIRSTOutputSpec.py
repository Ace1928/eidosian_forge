import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class ReplaceFSwithFIRSTOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')