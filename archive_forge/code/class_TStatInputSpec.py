import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class TStatInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dTstat', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_tstat', desc='output image file name', argstr='-prefix %s', name_source='in_file')
    mask = File(desc='mask file', argstr='-mask %s', exists=True)
    options = Str(desc='selected statistical output', argstr='%s')