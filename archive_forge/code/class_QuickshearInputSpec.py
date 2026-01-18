from .base import CommandLineInputSpec, CommandLine, traits, TraitedSpec, File
from ..external.due import BibTeX
class QuickshearInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, position=1, argstr='%s', mandatory=True, desc='neuroimage to deface')
    mask_file = File(exists=True, position=2, argstr='%s', desc='brain mask', mandatory=True)
    out_file = File(name_template='%s_defaced', name_source='in_file', position=3, argstr='%s', desc='defaced output image', keep_extension=True)
    buff = traits.Int(position=4, argstr='%d', desc='buffer size (in voxels) between shearing plane and the brain')