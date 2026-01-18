import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class VolisoInputSpec(CommandLineInputSpec):
    input_file = File(desc='input file to convert to isotropic sampling', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_voliso.mnc')
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='--verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='--clobber', usedefault=True, default_value=True)
    maxstep = traits.Float(desc='The target maximum step desired in the output volume.', argstr='--maxstep %s')
    minstep = traits.Float(desc='The target minimum step desired in the output volume.', argstr='--minstep %s')
    avgstep = traits.Bool(desc='Calculate the maximum step from the average steps of the input volume.', argstr='--avgstep')