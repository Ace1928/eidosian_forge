import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class ErodeInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='Input mask image to be eroded')
    out_filename = File(genfile=True, argstr='%s', position=-1, desc='Output image filename')
    number_of_passes = traits.Int(argstr='-npass %s', desc='the number of passes (default: 1)')
    dilate = traits.Bool(argstr='-dilate', position=1, desc='Perform dilation rather than erosion')
    quiet = traits.Bool(argstr='-quiet', position=1, desc='Do not display information messages or progress status.')
    debug = traits.Bool(argstr='-debug', position=1, desc='Display debugging messages.')