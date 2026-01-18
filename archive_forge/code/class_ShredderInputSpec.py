import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class ShredderInputSpec(StdOutCommandLineInputSpec):
    in_file = File(exists=True, argstr='< %s', mandatory=True, position=-2, desc='raw binary data file')
    offset = traits.Int(argstr='%d', units='NA', desc='initial offset of offset bytes', position=1)
    chunksize = traits.Int(argstr='%d', units='NA', desc='reads and outputs a chunk of chunksize bytes', position=2)
    space = traits.Int(argstr='%d', units='NA', desc='skips space bytes', position=3)