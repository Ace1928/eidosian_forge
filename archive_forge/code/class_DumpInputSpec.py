import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class DumpInputSpec(StdOutCommandLineInputSpec):
    input_file = File(desc='input file', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_dump.txt', keep_extension=False)
    _xor_coords_or_header = ('coordinate_data', 'header_data')
    coordinate_data = traits.Bool(desc='Coordinate variable data and header information.', argstr='-c', xor=_xor_coords_or_header)
    header_data = traits.Bool(desc='Header information only, no data.', argstr='-h', xor=_xor_coords_or_header)
    _xor_annotations = ('annotations_brief', 'annotations_full')
    annotations_brief = traits.Enum('c', 'f', argstr='-b %s', desc='Brief annotations for C or Fortran indices in data.', xor=_xor_annotations)
    annotations_full = traits.Enum('c', 'f', argstr='-f %s', desc='Full annotations for C or Fortran indices in data.', xor=_xor_annotations)
    variables = InputMultiPath(traits.Str, desc='Output data for specified variables only.', sep=',', argstr='-v %s')
    line_length = traits.Range(low=0, desc='Line length maximum in data section (default 80).', argstr='-l %d')
    netcdf_name = traits.Str(desc='Name for netCDF (default derived from file name).', argstr='-n %s')
    precision = traits.Either(traits.Int(), traits.Tuple(traits.Int, traits.Int), desc='Display floating-point values with less precision', argstr='%s')