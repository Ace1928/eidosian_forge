import os
from ...utils.filemanip import split_filename
from ..base import (
class ComputeFractionalAnisotropyOutputSpec(TraitedSpec):
    fa = File(exists=True, desc='Fractional Anisotropy Map')