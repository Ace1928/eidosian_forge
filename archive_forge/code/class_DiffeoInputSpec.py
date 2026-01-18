from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class DiffeoInputSpec(CommandLineInputSpec):
    fixed_file = File(desc='fixed tensor volume', exists=True, position=0, argstr='%s')
    moving_file = File(desc='moving tensor volume', exists=True, position=1, argstr='%s', copyfile=False)
    mask_file = File(desc='mask', exists=True, position=2, argstr='%s')
    legacy = traits.Enum(1, desc='legacy parameter; always set to 1', usedefault=True, mandatory=True, position=3, argstr='%d')
    n_iters = traits.Int(6, desc='number of iterations', mandatory=True, position=4, argstr='%d', usedefault=True)
    ftol = traits.Float(0.002, desc='iteration for the optimization to stop', mandatory=True, position=5, argstr='%g', usedefault=True)