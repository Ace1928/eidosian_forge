import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class CopyInputSpec(CommandLineInputSpec):
    input_file = File(desc='input file to copy', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_copy.mnc')
    _xor_pixel = ('pixel_values', 'real_values')
    pixel_values = traits.Bool(desc='Copy pixel values as is.', argstr='-pixel_values', xor=_xor_pixel)
    real_values = traits.Bool(desc='Copy real pixel intensities (default).', argstr='-real_values', xor=_xor_pixel)