import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FASTOutputSpec(TraitedSpec):
    """Specify possible outputs from FAST"""
    tissue_class_map = File(exists=True, desc='path/name of binary segmented volume file one val for each class  _seg')
    tissue_class_files = OutputMultiPath(File(desc='path/name of binary segmented volumes one file for each class  _seg_x'))
    restored_image = OutputMultiPath(File(desc='restored images (one for each input image) named according to the input images _restore'))
    mixeltype = File(desc='path/name of mixeltype volume file _mixeltype')
    partial_volume_map = File(desc='path/name of partial volume file _pveseg')
    partial_volume_files = OutputMultiPath(File(desc='path/name of partial volumes files one for each class, _pve_x'))
    bias_field = OutputMultiPath(File(desc='Estimated bias field _bias'))
    probability_maps = OutputMultiPath(File(desc='filenames, one for each class, for each input, prob_x'))