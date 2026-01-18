import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class NIfTIDT2CaminoOutputSpec(TraitedSpec):
    out_file = File(desc='diffusion tensors data in Camino format')