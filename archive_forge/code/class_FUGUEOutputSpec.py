import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FUGUEOutputSpec(TraitedSpec):
    unwarped_file = File(desc='unwarped file')
    warped_file = File(desc='forward warped file')
    shift_out_file = File(desc='voxel shift map file')
    fmap_out_file = File(desc='fieldmap file')