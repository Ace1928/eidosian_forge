import os
from functools import lru_cache
import numpy as np
from numpy import ones, kron, mean, eye, hstack, tile
from numpy.linalg import pinv
import nibabel as nb
from ..interfaces.base import (
class ICCOutputSpec(TraitedSpec):
    icc_map = File(exists=True)
    session_var_map = File(exists=True, desc='variance between sessions')
    subject_var_map = File(exists=True, desc='variance between subjects')