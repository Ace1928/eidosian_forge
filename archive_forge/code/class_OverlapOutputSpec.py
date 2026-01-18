import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
class OverlapOutputSpec(TraitedSpec):
    jaccard = traits.Float(desc='averaged jaccard index')
    dice = traits.Float(desc='averaged dice index')
    roi_ji = traits.List(traits.Float(), desc='the Jaccard index (JI) per ROI')
    roi_di = traits.List(traits.Float(), desc='the Dice index (DI) per ROI')
    volume_difference = traits.Float(desc='averaged volume difference')
    roi_voldiff = traits.List(traits.Float(), desc='volume differences of ROIs')
    labels = traits.List(traits.Int(), desc='detected labels')
    diff_file = File(exists=True, desc='error map of differences')