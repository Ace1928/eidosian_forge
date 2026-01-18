import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class DWIBiasCorrectOutputSpec(TraitedSpec):
    bias = File(desc='the output bias field', exists=True)
    out_file = File(desc='the output bias corrected DWI image', exists=True)