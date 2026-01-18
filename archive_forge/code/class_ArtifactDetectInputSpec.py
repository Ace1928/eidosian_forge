import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
class ArtifactDetectInputSpec(BaseInterfaceInputSpec):
    realigned_files = InputMultiPath(File(exists=True), desc='Names of realigned functional data files', mandatory=True)
    realignment_parameters = InputMultiPath(File(exists=True), mandatory=True, desc='Names of realignment parameters corresponding to the functional data files')
    parameter_source = traits.Enum('SPM', 'FSL', 'AFNI', 'NiPy', 'FSFAST', desc='Source of movement parameters', mandatory=True)
    use_differences = traits.ListBool([True, False], minlen=2, maxlen=2, usedefault=True, desc='Use differences between successive motion (first element) and intensity parameter (second element) estimates in order to determine outliers.  (default is [True, False])')
    use_norm = traits.Bool(True, usedefault=True, requires=['norm_threshold'], desc='Uses a composite of the motion parameters in order to determine outliers.')
    norm_threshold = traits.Float(xor=['rotation_threshold', 'translation_threshold'], mandatory=True, desc='Threshold to use to detect motion-related outliers when composite motion is being used')
    rotation_threshold = traits.Float(mandatory=True, xor=['norm_threshold'], desc='Threshold (in radians) to use to detect rotation-related outliers')
    translation_threshold = traits.Float(mandatory=True, xor=['norm_threshold'], desc='Threshold (in mm) to use to detect translation-related outliers')
    zintensity_threshold = traits.Float(mandatory=True, desc='Intensity Z-threshold use to detection images that deviate from the mean')
    mask_type = traits.Enum('spm_global', 'file', 'thresh', mandatory=True, desc='Type of mask that should be used to mask the functional data. *spm_global* uses an spm_global like calculation to determine the brain mask. *file* specifies a brain mask file (should be an image file consisting of 0s and 1s). *thresh* specifies a threshold to use. By default all voxels are used,unless one of these mask types are defined')
    mask_file = File(exists=True, desc="Mask file to be used if mask_type is 'file'.")
    mask_threshold = traits.Float(desc="Mask threshold to be used if mask_type is 'thresh'.")
    intersect_mask = traits.Bool(True, usedefault=True, desc='Intersect the masks when computed from spm_global.')
    save_plot = traits.Bool(True, desc='save plots containing outliers', usedefault=True)
    plot_type = traits.Enum('png', 'svg', 'eps', 'pdf', desc='file type of the outlier plot', usedefault=True)
    bound_by_brainmask = traits.Bool(False, desc='use the brain mask to determine bounding boxfor composite norm (worksfor SPM and Nipy - currentlyinaccurate for FSL, AFNI', usedefault=True)
    global_threshold = traits.Float(8.0, desc="use this threshold when mask type equal's spm_global", usedefault=True)