import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class QualityIndexInputSpec(CommandLineInputSpec):
    in_file = File(argstr='%s', mandatory=True, exists=True, position=-2, desc='input dataset')
    mask = File(exists=True, argstr='-mask %s', xor=['autoclip', 'automask'], desc='compute correlation only across masked voxels')
    spearman = traits.Bool(False, usedefault=True, argstr='-spearman', desc='Quality index is 1 minus the Spearman (rank) correlation coefficient of each sub-brick with the median sub-brick. (default).')
    quadrant = traits.Bool(False, usedefault=True, argstr='-quadrant', desc='Similar to -spearman, but using 1 minus the quadrant correlation coefficient as the quality index.')
    autoclip = traits.Bool(False, usedefault=True, argstr='-autoclip', xor=['mask'], desc='clip off small voxels')
    automask = traits.Bool(False, usedefault=True, argstr='-automask', xor=['mask'], desc='clip off small voxels')
    clip = traits.Float(argstr='-clip %f', desc='clip off values below')
    interval = traits.Bool(False, usedefault=True, argstr='-range', desc='write out the median + 3.5 MAD of outlier count with each timepoint')
    out_file = File(name_template='%s_tqual', name_source=['in_file'], argstr='> %s', keep_extension=False, position=-1, desc='capture standard output')