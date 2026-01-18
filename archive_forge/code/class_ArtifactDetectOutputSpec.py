import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
class ArtifactDetectOutputSpec(TraitedSpec):
    outlier_files = OutputMultiPath(File(exists=True), desc='One file for each functional run containing a list of 0-based indices corresponding to outlier volumes')
    intensity_files = OutputMultiPath(File(exists=True), desc='One file for each functional run containing the global intensity values determined from the brainmask')
    norm_files = OutputMultiPath(File, desc='One file for each functional run containing the composite norm')
    statistic_files = OutputMultiPath(File(exists=True), desc='One file for each functional run containing information about the different types of artifacts and if design info is provided then details of stimulus correlated motion and a listing or artifacts by event type.')
    plot_files = OutputMultiPath(File, desc='One image file for each functional run containing the detected outliers')
    mask_files = OutputMultiPath(File, desc='One image file for each functional run containing the mask used for global signal calculation')
    displacement_files = OutputMultiPath(File, desc='One image file for each functional run containing the voxel displacement timeseries')