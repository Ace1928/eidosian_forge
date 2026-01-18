import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class ToRawInputSpec(StdOutCommandLineInputSpec):
    input_file = File(desc='input file', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', position=-1, name_source=['input_file'], hash_files=False, name_template='%s.raw', keep_extension=False)
    _xor_write = ('write_byte', 'write_short', 'write_int', 'write_long', 'write_float', 'write_double')
    write_byte = traits.Bool(desc='Write out data as bytes.', argstr='-byte', xor=_xor_write)
    write_short = traits.Bool(desc='Write out data as short integers.', argstr='-short', xor=_xor_write)
    write_int = traits.Bool(desc='Write out data as 32-bit integers.', argstr='-int', xor=_xor_write)
    write_long = traits.Bool(desc='Superseded by write_int.', argstr='-long', xor=_xor_write)
    write_float = traits.Bool(desc='Write out data as single precision floating-point values.', argstr='-float', xor=_xor_write)
    write_double = traits.Bool(desc='Write out data as double precision floating-point values.', argstr='-double', xor=_xor_write)
    _xor_signed = ('write_signed', 'write_unsigned')
    write_signed = traits.Bool(desc='Write out signed data.', argstr='-signed', xor=_xor_signed)
    write_unsigned = traits.Bool(desc='Write out unsigned data.', argstr='-unsigned', xor=_xor_signed)
    write_range = traits.Tuple(traits.Float, traits.Float, argstr='-range %s %s', desc='Specify the range of output values.Default value: 1.79769e+308 1.79769e+308.')
    _xor_normalize = ('normalize', 'nonormalize')
    normalize = traits.Bool(desc='Normalize integer pixel values to file max and min.', argstr='-normalize', xor=_xor_normalize)
    nonormalize = traits.Bool(desc='Turn off pixel normalization.', argstr='-nonormalize', xor=_xor_normalize)