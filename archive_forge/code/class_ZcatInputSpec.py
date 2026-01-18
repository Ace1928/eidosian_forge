import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class ZcatInputSpec(AFNICommandInputSpec):
    in_files = InputMultiPath(File(desc='input files to 3dZcat', exists=True), argstr='%s', position=-1, mandatory=True, copyfile=False)
    out_file = File(name_template='%s_zcat', desc="output dataset prefix name (default 'zcat')", argstr='-prefix %s', name_source='in_files')
    datum = traits.Enum('byte', 'short', 'float', argstr='-datum %s', desc="specify data type for output. Valid types are 'byte', 'short' and 'float'.")
    verb = traits.Bool(desc='print out some verbositiness as the program proceeds.', argstr='-verb')
    fscale = traits.Bool(desc='Force scaling of the output to the maximum integer range.  This only has effect if the output datum is byte or short (either forced or defaulted). This option is sometimes necessary to eliminate unpleasant truncation artifacts.', argstr='-fscale', xor=['nscale'])
    nscale = traits.Bool(desc="Don't do any scaling on output to byte or short datasets. This may be especially useful when operating on mask datasets whose output values are only 0's and 1's.", argstr='-nscale', xor=['fscale'])