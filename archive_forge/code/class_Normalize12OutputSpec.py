import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class Normalize12OutputSpec(TraitedSpec):
    deformation_field = OutputMultiPath(File(exists=True), desc='NIfTI file containing 3 deformation fields for the deformation in x, y and z dimension')
    normalized_image = OutputMultiPath(File(exists=True), desc='Normalized file that needed to be aligned')
    normalized_files = OutputMultiPath(File(exists=True), desc='Normalized other files')