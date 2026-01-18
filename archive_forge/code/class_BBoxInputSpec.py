import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BBoxInputSpec(StdOutCommandLineInputSpec):
    input_file = File(desc='input file', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file containing bounding box corners', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_bbox.txt', keep_extension=False)
    threshold = traits.Int(0, desc='VIO_Real value threshold for bounding box. Default value: 0.', argstr='-threshold')
    _xor_one_two = ('one_line', 'two_lines')
    one_line = traits.Bool(desc='Output on one line (default): start_x y z width_x y z', argstr='-one_line', xor=_xor_one_two)
    two_lines = traits.Bool(desc='Write output with two rows (start and width).', argstr='-two_lines', xor=_xor_one_two)
    format_mincresample = traits.Bool(desc='Output format for mincresample: (-step x y z -start x y z -nelements x y z', argstr='-mincresample')
    format_mincreshape = traits.Bool(desc='Output format for mincreshape: (-start x,y,z -count dx,dy,dz', argstr='-mincreshape')
    format_minccrop = traits.Bool(desc='Output format for minccrop: (-xlim x1 x2 -ylim y1 y2 -zlim z1 z2', argstr='-minccrop')