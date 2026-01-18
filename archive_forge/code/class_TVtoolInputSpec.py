from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class TVtoolInputSpec(CommandLineInputSpec):
    in_file = File(desc='scalar volume to resample', exists=True, argstr='-in %s', mandatory=True)
    'NOTE: there are a lot more options here; not implementing all of them'
    in_flag = traits.Enum('fa', 'tr', 'ad', 'rd', 'pd', 'rgb', argstr='-%s', desc='')
    out_file = File(argstr='-out %s', genfile=True)