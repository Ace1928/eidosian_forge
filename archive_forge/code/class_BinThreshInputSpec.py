from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class BinThreshInputSpec(CommandLineInputSpec):
    in_file = File(desc='Image to threshold/binarize', exists=True, position=0, argstr='%s', mandatory=True)
    out_file = File(desc='output path', position=1, argstr='%s', keep_extension=True, name_source='in_file', name_template='%s_thrbin')
    lower_bound = traits.Float(0.01, usedefault=True, position=2, argstr='%g', mandatory=True, desc='lower bound of binarization range')
    upper_bound = traits.Float(100, usedefault=True, position=3, argstr='%g', mandatory=True, desc='upper bound of binarization range')
    inside_value = traits.Float(1, position=4, argstr='%g', usedefault=True, mandatory=True, desc='value for voxels in binarization range')
    outside_value = traits.Float(0, position=5, argstr='%g', usedefault=True, mandatory=True, desc='value for voxelsoutside of binarization range')