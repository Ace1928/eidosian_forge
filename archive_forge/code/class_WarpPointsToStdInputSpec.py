import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class WarpPointsToStdInputSpec(WarpPointsBaseInputSpec):
    img_file = File(exists=True, argstr='-img %s', mandatory=True, desc='filename of input image')
    std_file = File(exists=True, argstr='-std %s', mandatory=True, desc='filename of destination image')
    premat_file = File(exists=True, argstr='-premat %s', desc='filename of pre-warp affine transform (e.g. example_func2highres.mat)')