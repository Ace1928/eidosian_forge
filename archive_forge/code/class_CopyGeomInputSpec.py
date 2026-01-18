import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class CopyGeomInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0, desc='source image')
    dest_file = File(exists=True, mandatory=True, argstr='%s', position=1, desc='destination image', copyfile=True, output_name='out_file', name_source='dest_file', name_template='%s')
    ignore_dims = traits.Bool(desc='Do not copy image dimensions', argstr='-d', position='-1')