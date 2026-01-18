import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class EvalInputSpec(AFNICommandInputSpec):
    in_file_a = File(desc='input file to 1deval', argstr='-a %s', position=0, mandatory=True, exists=True)
    in_file_b = File(desc='operand file to 1deval', argstr='-b %s', position=1, exists=True)
    in_file_c = File(desc='operand file to 1deval', argstr='-c %s', position=2, exists=True)
    out_file = File(name_template='%s_calc', desc='output image file name', argstr='-prefix %s', name_source='in_file_a')
    out1D = traits.Bool(desc='output in 1D', argstr='-1D')
    expr = Str(desc='expr', argstr='-expr "%s"', position=3, mandatory=True)
    start_idx = traits.Int(desc='start index for in_file_a', requires=['stop_idx'])
    stop_idx = traits.Int(desc='stop index for in_file_a', requires=['start_idx'])
    single_idx = traits.Int(desc='volume index for in_file_a')
    other = File(desc='other options', argstr='')