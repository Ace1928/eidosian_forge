import os
from ...utils.filemanip import split_filename
from ..base import (
class ComputeTensorTraceOutputSpec(TraitedSpec):
    trace = File(exists=True, desc='Trace of the diffusion tensor')