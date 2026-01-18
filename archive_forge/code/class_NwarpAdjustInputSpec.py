import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class NwarpAdjustInputSpec(AFNICommandInputSpec):
    warps = InputMultiPath(File(exists=True), minlen=5, mandatory=True, argstr='-nwarp %s', desc='List of input 3D warp datasets')
    in_files = InputMultiPath(File(exists=True), minlen=5, argstr='-source %s', desc='List of input 3D datasets to be warped by the adjusted warp datasets.  There must be exactly as many of these datasets as there are input warps.')
    out_file = File(desc='Output mean dataset, only needed if in_files are also given. The output dataset will be on the common grid shared by the source datasets.', argstr='-prefix %s', name_source='in_files', name_template='%s_NwarpAdjust', keep_extension=True, requires=['in_files'])