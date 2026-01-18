import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BigAverageInputSpec(CommandLineInputSpec):
    input_files = InputMultiPath(File(exists=True), desc='input file(s)', mandatory=True, sep=' ', argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_files'], hash_files=False, name_template='%s_bigaverage.mnc')
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='--verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='--clobber', usedefault=True, default_value=True)
    output_float = traits.Bool(desc='Output files with float precision.', argstr='--float')
    robust = traits.Bool(desc='Perform robust averaging, features that are outside 1 standarddeviation from the mean are downweighted. Works well for noisydata with artifacts. see the --tmpdir option if you have alarge number of input files.', argstr='-robust')
    tmpdir = Directory(desc='temporary files directory', argstr='-tmpdir %s')
    sd_file = File(desc='Place standard deviation image in specified file.', argstr='--sdfile %s', name_source=['input_files'], hash_files=False, name_template='%s_bigaverage_stdev.mnc')