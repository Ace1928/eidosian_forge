import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class MRTrixViewerInputSpec(CommandLineInputSpec):
    in_files = InputMultiPath(File(exists=True), argstr='%s', mandatory=True, position=-2, desc='Input images to be viewed')
    quiet = traits.Bool(argstr='-quiet', position=1, desc='Do not display information messages or progress status.')
    debug = traits.Bool(argstr='-debug', position=1, desc='Display debugging messages.')