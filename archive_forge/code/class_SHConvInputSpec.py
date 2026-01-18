import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class SHConvInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3, desc='input ODF image')
    response = File(exists=True, mandatory=True, argstr='%s', position=-2, desc='The response function')
    out_file = File(name_template='%s_shconv.mif', name_source=['in_file'], argstr='%s', position=-1, usedefault=True, desc='the output spherical harmonics')