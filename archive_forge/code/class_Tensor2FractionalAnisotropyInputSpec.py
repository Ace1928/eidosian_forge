import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class Tensor2FractionalAnisotropyInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='Diffusion tensor image')
    out_filename = File(genfile=True, argstr='%s', position=-1, desc='Output Fractional Anisotropy filename')
    quiet = traits.Bool(argstr='-quiet', position=1, desc='Do not display information messages or progress status.')
    debug = traits.Bool(argstr='-debug', position=1, desc='Display debugging messages.')