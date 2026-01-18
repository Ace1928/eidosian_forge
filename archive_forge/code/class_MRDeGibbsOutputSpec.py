import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class MRDeGibbsOutputSpec(TraitedSpec):
    out_file = File(desc='the output unringed DWI image', exists=True)