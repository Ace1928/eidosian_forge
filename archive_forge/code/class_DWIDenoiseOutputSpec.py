import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class DWIDenoiseOutputSpec(TraitedSpec):
    noise = File(desc='the output noise map', exists=True)
    out_file = File(desc='the output denoised DWI image', exists=True)