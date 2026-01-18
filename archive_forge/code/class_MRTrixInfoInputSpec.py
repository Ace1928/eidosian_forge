import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class MRTrixInfoInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='Input images to be read')