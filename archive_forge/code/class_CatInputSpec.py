import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class CatInputSpec(AFNICommandInputSpec):
    in_files = traits.List(File(exists=True), argstr='%s', mandatory=True, position=-2)
    out_file = File(argstr='> %s', value='catout.1d', usedefault=True, desc='output (concatenated) file name', position=-1, mandatory=True)
    omitconst = traits.Bool(desc='Omit columns that are identically constant from output.', argstr='-nonconst')
    keepfree = traits.Bool(desc="Keep only columns that are marked as 'free' in the 3dAllineate header from '-1Dparam_save'. If there is no such header, all columns are kept.", argstr='-nonfixed')
    out_format = traits.Enum('int', 'nice', 'double', 'fint', 'cint', argstr='-form %s', desc='specify data type for output.', xor=['out_int', 'out_nice', 'out_double', 'out_fint', 'out_cint'])
    stack = traits.Bool(desc='Stack the columns of the resultant matrix in the output.', argstr='-stack')
    sel = traits.Str(desc='Apply the same column/row selection string to all filenames on the command line.', argstr='-sel %s')
    out_int = traits.Bool(desc='specify int data type for output', argstr='-i', xor=['out_format', 'out_nice', 'out_double', 'out_fint', 'out_cint'])
    out_nice = traits.Bool(desc='specify nice data type for output', argstr='-n', xor=['out_format', 'out_int', 'out_double', 'out_fint', 'out_cint'])
    out_double = traits.Bool(desc='specify double data type for output', argstr='-d', xor=['out_format', 'out_nice', 'out_int', 'out_fint', 'out_cint'])
    out_fint = traits.Bool(desc='specify int, rounded down, data type for output', argstr='-f', xor=['out_format', 'out_nice', 'out_double', 'out_int', 'out_cint'])
    out_cint = traits.Bool(desc='specify int, rounded up, data type for output', xor=['out_format', 'out_nice', 'out_double', 'out_fint', 'out_int'])