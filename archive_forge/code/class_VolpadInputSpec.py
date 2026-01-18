import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class VolpadInputSpec(CommandLineInputSpec):
    """
    Not implemented:

    -fake         do a dry run, (echo cmds only)
    -nofake       opposite of -fake [default]

     | volpad pads a MINC volume
     |
     | Problems or comments should be sent to: a.janke@gmail.com

    Summary of options:

    -- General Options -------------------------------------------------------------
       -verbose          be verbose
       -noverbose        opposite of -verbose [default]
       -clobber          clobber existing files
       -noclobber        opposite of -clobber [default]
       -fake             do a dry run, (echo cmds only)
       -nofake           opposite of -fake [default]


    """
    input_file = File(desc='input file to centre', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_volpad.mnc')
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='-verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    auto = traits.Bool(desc='Automatically determine padding distances (uses -distance as max). Default: False.', argstr='-auto')
    auto_freq = traits.Float(desc='Frequency of voxels over bimodalt threshold to stop at [default: 500].', argstr='-auto_freq %s')
    distance = traits.Int(desc='Padding distance (in voxels) [default: 4].', argstr='-distance %s')
    smooth = traits.Bool(desc='Smooth (blur) edges before padding. Default: False.', argstr='-smooth')
    smooth_distance = traits.Int(desc='Smoothing distance (in voxels) [default: 4].', argstr='-smooth_distance %s')