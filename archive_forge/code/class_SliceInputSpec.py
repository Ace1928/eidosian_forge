import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class SliceInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr='%s', position=0, mandatory=True, desc='input filename', copyfile=False)
    out_base_name = traits.Str(argstr='%s', position=1, desc='outputs prefix')