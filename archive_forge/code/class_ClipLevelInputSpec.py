import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class ClipLevelInputSpec(CommandLineInputSpec):
    in_file = File(desc='input file to 3dClipLevel', argstr='%s', position=-1, mandatory=True, exists=True)
    mfrac = traits.Float(desc='Use the number ff instead of 0.50 in the algorithm', argstr='-mfrac %s', position=2)
    doall = traits.Bool(desc='Apply the algorithm to each sub-brick separately.', argstr='-doall', position=3, xor='grad')
    grad = File(desc="Also compute a 'gradual' clip level as a function of voxel position, and output that to a dataset.", argstr='-grad %s', position=3, xor='doall')