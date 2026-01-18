import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class DespikeInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dDespike', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_despike', desc='output image file name', argstr='-prefix %s', name_source='in_file')