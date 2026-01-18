from ..base import CommandLineInputSpec, CommandLine, TraitedSpec, File
class VtoMatInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='-in %s', mandatory=True, position=1, desc='in file')
    out_file = File(name_template='%s.mat', keep_extension=False, argstr='-out %s', hash_files=False, position=-1, desc='output mat file', name_source=['in_file'])