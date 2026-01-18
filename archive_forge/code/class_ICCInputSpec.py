import os
from functools import lru_cache
import numpy as np
from numpy import ones, kron, mean, eye, hstack, tile
from numpy.linalg import pinv
import nibabel as nb
from ..interfaces.base import (
class ICCInputSpec(BaseInterfaceInputSpec):
    subjects_sessions = traits.List(traits.List(File(exists=True)), desc='n subjects m sessions 3D stat files', mandatory=True)
    mask = File(exists=True, mandatory=True)