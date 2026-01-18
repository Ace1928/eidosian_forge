import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
def _gen_thresholded_map_filename(self):
    _, fname, ext = split_filename(self.inputs.stat_image)
    return os.path.abspath(fname + '_thr' + ext)