import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BestLinRegInputSpec(CommandLineInputSpec):
    source = File(desc='source Minc file', exists=True, mandatory=True, argstr='%s', position=-4)
    target = File(desc='target Minc file', exists=True, mandatory=True, argstr='%s', position=-3)
    output_xfm = File(desc='output xfm file', genfile=True, argstr='%s', position=-2, name_source=['source'], hash_files=False, name_template='%s_bestlinreg.xfm', keep_extension=False)
    output_mnc = File(desc='output mnc file', genfile=True, argstr='%s', position=-1, name_source=['source'], hash_files=False, name_template='%s_bestlinreg.mnc', keep_extension=False)
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='-verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    '\n    -init_xfm     initial transformation (default identity)\n    -source_mask  source mask to use during fitting\n    -target_mask  target mask to use during fitting\n    -lsq9         use 9-parameter transformation (default)\n    -lsq12        use 12-parameter transformation (default -lsq9)\n    -lsq6         use 6-parameter transformation\n    '