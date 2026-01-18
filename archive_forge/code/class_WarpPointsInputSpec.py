import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class WarpPointsInputSpec(WarpPointsBaseInputSpec):
    src_file = File(exists=True, argstr='-src %s', mandatory=True, desc='filename of source image')
    dest_file = File(exists=True, argstr='-dest %s', mandatory=True, desc='filename of destination image')