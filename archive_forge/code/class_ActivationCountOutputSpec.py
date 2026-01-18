import os
import nibabel as nb
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import split_filename
class ActivationCountOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output activation count map')
    acm_pos = File(exists=True, desc='positive activation count map')
    acm_neg = File(exists=True, desc='negative activation count map')