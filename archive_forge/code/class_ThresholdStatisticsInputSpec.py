import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class ThresholdStatisticsInputSpec(SPMCommandInputSpec):
    spm_mat_file = File(exists=True, desc='absolute path to SPM.mat', copyfile=True, mandatory=True)
    stat_image = File(exists=True, desc='stat image', copyfile=False, mandatory=True)
    contrast_index = traits.Int(mandatory=True, desc='which contrast in the SPM.mat to use')
    height_threshold = traits.Float(desc='stat value for initial thresholding (defining clusters)', mandatory=True)
    extent_threshold = traits.Int(0, usedefault=True, desc='Minimum cluster size in voxels')