import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class ReshapeInputSpec(CommandLineInputSpec):
    input_file = File(desc='input file', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_reshape.mnc')
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='-verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    write_short = traits.Bool(desc='Convert to short integer data.', argstr='-short')