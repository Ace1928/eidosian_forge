import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class LFCDInputSpec(CentralityInputSpec):
    """LFCD inputspec"""
    in_file = File(desc='input file to 3dLFCD', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)