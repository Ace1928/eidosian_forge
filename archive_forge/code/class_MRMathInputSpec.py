import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class MRMathInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3, desc='input image')
    out_file = File(argstr='%s', mandatory=True, position=-1, desc='output image')
    operation = traits.Enum('mean', 'median', 'sum', 'product', 'rms', 'norm', 'var', 'std', 'min', 'max', 'absmax', 'magmax', argstr='%s', position=-2, mandatory=True, desc='operation to computer along a specified axis')
    axis = traits.Int(0, argstr='-axis %d', desc='specified axis to perform the operation along')