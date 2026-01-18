import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class XfmInvertInputSpec(CommandLineInputSpec):
    input_file = File(desc='input file', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1)
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='-verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)