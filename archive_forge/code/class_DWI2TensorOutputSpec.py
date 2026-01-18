import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class DWI2TensorOutputSpec(TraitedSpec):
    tensor = File(exists=True, desc='path/name of output diffusion tensor image')