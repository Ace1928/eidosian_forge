import os
from .base import (
class Bru2InputSpec(CommandLineInputSpec):
    input_dir = Directory(desc='Input Directory', exists=True, mandatory=True, position=-1, argstr='%s')
    actual_size = traits.Bool(argstr='-a', desc='Keep actual size - otherwise x10 scale so animals match human.')
    force_conversion = traits.Bool(argstr='-f', desc='Force conversion of localizers images (multiple slice orientations).')
    compress = traits.Bool(argstr='-z', desc='gz compress images (".nii.gz").')
    append_protocol_name = traits.Bool(argstr='-p', desc='Append protocol name to output filename.')
    output_filename = traits.Str(argstr='-o %s', desc='Output filename (".nii" will be appended, or ".nii.gz" if the "-z" compress option is selected)', genfile=True)