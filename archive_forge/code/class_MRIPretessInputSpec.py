import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIPretessInputSpec(FSTraitedSpec):
    in_filled = File(exists=True, mandatory=True, position=-4, argstr='%s', desc='filled volume, usually wm.mgz')
    label = traits.Either(traits.Str('wm'), traits.Int(1), argstr='%s', default='wm', mandatory=True, usedefault=True, position=-3, desc="label to be picked up, can be a Freesurfer's string like 'wm' or a label value (e.g. 127 for rh or 255 for lh)")
    in_norm = File(exists=True, mandatory=True, position=-2, argstr='%s', desc='the normalized, brain-extracted T1w image. Usually norm.mgz')
    out_file = File(position=-1, argstr='%s', name_source=['in_filled'], name_template='%s_pretesswm', keep_extension=True, desc='the output file after mri_pretess.')
    nocorners = traits.Bool(False, argstr='-nocorners', desc='do not remove corner configurations in addition to edge ones.')
    keep = traits.Bool(False, argstr='-keep', desc='keep WM edits')
    test = traits.Bool(False, argstr='-test', desc='adds a voxel that should be removed by mri_pretess. The value of the voxel is set to that of an ON-edited WM, so it should be kept with -keep. The output will NOT be saved.')