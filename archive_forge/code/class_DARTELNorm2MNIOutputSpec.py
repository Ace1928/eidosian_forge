import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class DARTELNorm2MNIOutputSpec(TraitedSpec):
    normalized_files = OutputMultiPath(File(exists=True), desc='Normalized files in MNI space')
    normalization_parameter_file = File(exists=True, desc='Transform parameters to MNI space')