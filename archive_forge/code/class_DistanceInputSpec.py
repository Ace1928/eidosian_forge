import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
class DistanceInputSpec(BaseInterfaceInputSpec):
    volume1 = File(exists=True, mandatory=True, desc='Has to have the same dimensions as volume2.')
    volume2 = File(exists=True, mandatory=True, desc='Has to have the same dimensions as volume1.')
    method = traits.Enum('eucl_min', 'eucl_cog', 'eucl_mean', 'eucl_wmean', 'eucl_max', desc='""eucl_min": Euclidean distance between two closest points        "eucl_cog": mean Euclidean distance between the Center of Gravity        of volume1 and CoGs of volume2        "eucl_mean": mean Euclidean minimum distance of all volume2 voxels        to volume1        "eucl_wmean": mean Euclidean minimum distance of all volume2 voxels        to volume1 weighted by their values        "eucl_max": maximum over minimum Euclidean distances of all volume2        voxels to volume1 (also known as the Hausdorff distance)', usedefault=True)
    mask_volume = File(exists=True, desc='calculate overlap only within this mask.')