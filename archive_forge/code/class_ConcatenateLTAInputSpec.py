import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class ConcatenateLTAInputSpec(FSTraitedSpec):
    in_lta1 = File(exists=True, mandatory=True, argstr='%s', position=-3, desc='maps some src1 to dst1')
    in_lta2 = traits.Either(File(exists=True), 'identity.nofile', argstr='%s', position=-2, mandatory=True, desc='maps dst1(src2) to dst2')
    out_file = File(position=-1, argstr='%s', hash_files=False, name_source=['in_lta1'], name_template='%s_concat', keep_extension=True, desc='the combined LTA maps: src1 to dst2 = LTA2*LTA1')
    invert_1 = traits.Bool(argstr='-invert1', desc='invert in_lta1 before applying it')
    invert_2 = traits.Bool(argstr='-invert2', desc='invert in_lta2 before applying it')
    invert_out = traits.Bool(argstr='-invertout', desc='invert output LTA')
    out_type = traits.Enum('VOX2VOX', 'RAS2RAS', argstr='-out_type %d', desc='set final LTA type')
    tal_source_file = File(exists=True, argstr='-tal %s', position=-5, requires=['tal_template_file'], desc='if in_lta2 is talairach.xfm, specify source for talairach')
    tal_template_file = File(exists=True, argstr='%s', position=-4, requires=['tal_source_file'], desc='if in_lta2 is talairach.xfm, specify template for talairach')
    subject = traits.Str(argstr='-subject %s', desc='set subject in output LTA')