from ..base import CommandLineInputSpec, CommandLine, TraitedSpec, File
class Vnifti2ImageInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='-in %s', mandatory=True, position=1, desc='in file')
    attributes = File(exists=True, argstr='-attr %s', position=2, desc='attribute file')
    out_file = File(name_template='%s.v', keep_extension=False, argstr='-out %s', hash_files=False, position=-1, desc='output data file', name_source=['in_file'])