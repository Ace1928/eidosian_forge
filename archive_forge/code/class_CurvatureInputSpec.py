import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class CurvatureInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', position=-2, mandatory=True, exists=True, copyfile=True, desc='Input file for Curvature')
    threshold = traits.Float(argstr='-thresh %.3f', desc='Undocumented input threshold')
    n = traits.Bool(argstr='-n', desc='Undocumented boolean flag')
    averages = traits.Int(argstr='-a %d', desc='Perform this number iterative averages of curvature measure before saving')
    save = traits.Bool(argstr='-w', desc='Save curvature files (will only generate screen output without this option)')
    distances = traits.Tuple(traits.Int, traits.Int, argstr='-distances %d %d', desc='Undocumented input integer distances')
    copy_input = traits.Bool(desc='Copy input file to current directory')