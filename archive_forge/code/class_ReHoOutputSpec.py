import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class ReHoOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Voxelwise regional homogeneity map')
    out_vals = File(desc='Table of labelwise regional homogeneity values')