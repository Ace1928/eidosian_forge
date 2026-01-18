import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
class FuzzyOverlapInputSpec(BaseInterfaceInputSpec):
    in_ref = InputMultiPath(File(exists=True), mandatory=True, desc='Reference image. Requires the same dimensions as in_tst.')
    in_tst = InputMultiPath(File(exists=True), mandatory=True, desc='Test image. Requires the same dimensions as in_ref.')
    in_mask = File(exists=True, desc='calculate overlap only within mask')
    weighting = traits.Enum('none', 'volume', 'squared_vol', usedefault=True, desc="'none': no class-overlap weighting is performed. 'volume': computed class-overlaps are weighted by class volume 'squared_vol': computed class-overlaps are weighted by the squared volume of the class")
    out_file = File('diff.nii', desc='alternative name for resulting difference-map', usedefault=True)