from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class SVResampleInputSpec(CommandLineInputSpec):
    in_file = File(desc='image to resample', exists=True, mandatory=True, argstr='-in %s')
    out_file = File(desc='output path', name_source='in_file', name_template='%s_resampled', keep_extension=True, argstr='-out %s')
    target_file = File(desc='specs read from the target volume', argstr='-target %s', xor=['array_size', 'voxel_size', 'origin'])
    align = traits.Enum('center', 'origin', argstr='-align %s', desc='how to align output volume to input volume')
    array_size = traits.Tuple((traits.Int(), traits.Int(), traits.Int()), desc='resampled array size', xor=['target_file'], argstr='-size %d %d %d')
    voxel_size = traits.Tuple((traits.Float(), traits.Float(), traits.Float()), desc='resampled voxel size', xor=['target_file'], argstr='-vsize %g %g %g')
    origin = traits.Tuple((traits.Float(), traits.Float(), traits.Float()), desc='xyz origin', xor=['target_file'], argstr='-origin %g %g %g')