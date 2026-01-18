import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class PickAtlasInputSpec(BaseInterfaceInputSpec):
    atlas = File(exists=True, desc='Location of the atlas that will be used.', mandatory=True)
    labels = traits.Either(traits.Int, traits.List(traits.Int), desc='Labels of regions that will be included in the mask. Must be        compatible with the atlas used.', mandatory=True)
    hemi = traits.Enum('both', 'left', 'right', desc='Restrict the mask to only one hemisphere: left or right', usedefault=True)
    dilation_size = traits.Int(usedefault=True, desc='Defines how much the mask will be dilated (expanded in 3D).')
    output_file = File(desc='Where to store the output mask.')