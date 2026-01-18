import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class Reorient2StdInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s')
    out_file = File(genfile=True, hash_files=False, argstr='%s')