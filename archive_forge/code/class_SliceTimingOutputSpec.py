import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class SliceTimingOutputSpec(TraitedSpec):
    timecorrected_files = OutputMultiPath(traits.Either(traits.List(File(exists=True)), File(exists=True)), desc='slice time corrected files')