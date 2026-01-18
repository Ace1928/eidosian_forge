import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class EstimateContrastOutputSpec(TraitedSpec):
    con_images = OutputMultiPath(File(exists=True), desc='contrast images from a t-contrast')
    spmT_images = OutputMultiPath(File(exists=True), desc='stat images from a t-contrast')
    ess_images = OutputMultiPath(File(exists=True), desc='contrast images from an F-contrast')
    spmF_images = OutputMultiPath(File(exists=True), desc='stat images from an F-contrast')
    spm_mat_file = File(exists=True, desc='Updated SPM mat file')