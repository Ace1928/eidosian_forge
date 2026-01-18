import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class ThresholdInputSpec(SPMCommandInputSpec):
    spm_mat_file = File(exists=True, desc='absolute path to SPM.mat', copyfile=True, mandatory=True)
    stat_image = File(exists=True, desc='stat image', copyfile=False, mandatory=True)
    contrast_index = traits.Int(mandatory=True, desc='which contrast in the SPM.mat to use')
    use_fwe_correction = traits.Bool(True, usedefault=True, desc='whether to use FWE (Bonferroni) correction for initial threshold (height_threshold_type has to be set to p-value)')
    use_topo_fdr = traits.Bool(True, usedefault=True, desc='whether to use FDR over cluster extent probabilities')
    height_threshold = traits.Float(0.05, usedefault=True, desc='value for initial thresholding (defining clusters)')
    height_threshold_type = traits.Enum('p-value', 'stat', usedefault=True, desc='Is the cluster forming threshold a stat value or p-value?')
    extent_fdr_p_threshold = traits.Float(0.05, usedefault=True, desc='p threshold on FDR corrected cluster size probabilities')
    extent_threshold = traits.Int(0, usedefault=True, desc='Minimum cluster size in voxels')
    force_activation = traits.Bool(False, usedefault=True, desc='In case no clusters survive the topological inference step this will pick a culster with the highest sum of t-values. Use with care.')