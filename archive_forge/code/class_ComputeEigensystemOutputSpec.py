import os
from ...utils.filemanip import split_filename
from ..base import (
class ComputeEigensystemOutputSpec(TraitedSpec):
    eigen = File(exists=True, desc='Trace of the diffusion tensor')