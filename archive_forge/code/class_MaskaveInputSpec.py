import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class MaskaveInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dmaskave', argstr='%s', position=-2, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_maskave.1D', desc='output image file name', keep_extension=True, argstr='> %s', name_source='in_file', position=-1)
    mask = File(desc='matrix to align input file', argstr='-mask %s', position=1, exists=True)
    quiet = traits.Bool(desc='matrix to align input file', argstr='-quiet', position=2)