import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class SwapDimensionsInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position='1', desc='input image')
    _dims = ['x', '-x', 'y', '-y', 'z', '-z', 'RL', 'LR', 'AP', 'PA', 'IS', 'SI']
    new_dims = traits.Tuple(traits.Enum(_dims), traits.Enum(_dims), traits.Enum(_dims), argstr='%s %s %s', mandatory=True, desc='3-tuple of new dimension order')
    out_file = File(genfile=True, argstr='%s', desc='image to write', hash_files=False)