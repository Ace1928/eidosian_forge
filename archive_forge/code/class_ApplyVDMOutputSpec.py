import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class ApplyVDMOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(traits.Either(traits.List(File(exists=True)), File(exists=True)), desc='These will be the fieldmap corrected files.')
    mean_image = File(exists=True, desc='Mean image')