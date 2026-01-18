import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class GennlxfmInputSpec(CommandLineInputSpec):
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['like'], hash_files=False, name_template='%s_gennlxfm.xfm')
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='-verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    ident = traits.Bool(desc='Generate an identity xfm. Default: False.', argstr='-ident')
    step = traits.Int(desc='Output ident xfm step [default: 1].', argstr='-step %s')
    like = File(desc='Generate a nlxfm like this file.', exists=True, argstr='-like %s')