import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ConvertXFMInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=-1, desc='input transformation matrix')
    in_file2 = File(exists=True, argstr='%s', position=-2, desc='second input matrix (for use with fix_scale_skew or concat_xfm)')
    _options = ['invert_xfm', 'concat_xfm', 'fix_scale_skew']
    invert_xfm = traits.Bool(argstr='-inverse', position=-3, xor=_options, desc='invert input transformation')
    concat_xfm = traits.Bool(argstr='-concat', position=-3, xor=_options, requires=['in_file2'], desc='write joint transformation of two input matrices')
    fix_scale_skew = traits.Bool(argstr='-fixscaleskew', position=-3, xor=_options, requires=['in_file2'], desc='use secondary matrix to fix scale and skew')
    out_file = File(genfile=True, argstr='-omat %s', position=1, desc='final transformation matrix', hash_files=False)