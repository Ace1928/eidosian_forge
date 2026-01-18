import os
from ...utils.filemanip import split_filename
from ..base import (
class ComputeMeanDiffusivityOutputSpec(TraitedSpec):
    md = File(exists=True, desc='Mean Diffusivity Map')