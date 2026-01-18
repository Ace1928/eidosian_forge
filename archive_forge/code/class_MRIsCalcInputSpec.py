import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsCalcInputSpec(FSTraitedSpec):
    in_file1 = File(argstr='%s', position=-3, mandatory=True, exists=True, desc='Input file 1')
    action = traits.String(argstr='%s', position=-2, mandatory=True, desc='Action to perform on input file(s)')
    out_file = File(argstr='-o %s', mandatory=True, desc='Output file after calculation')
    in_file2 = File(argstr='%s', exists=True, position=-1, xor=['in_float', 'in_int'], desc='Input file 2')
    in_float = traits.Float(argstr='%f', position=-1, xor=['in_file2', 'in_int'], desc='Input float')
    in_int = traits.Int(argstr='%d', position=-1, xor=['in_file2', 'in_float'], desc='Input integer')