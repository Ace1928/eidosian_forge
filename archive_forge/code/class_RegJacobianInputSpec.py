import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegJacobianInputSpec(NiftyRegCommandInputSpec):
    """Input Spec for RegJacobian."""
    desc = 'Reference/target file (required if specifying CPP transformations.'
    ref_file = File(exists=True, desc=desc, argstr='-ref %s')
    trans_file = File(exists=True, desc='The input non-rigid transformation', argstr='-trans %s', mandatory=True)
    type = traits.Enum('jac', 'jacL', 'jacM', usedefault=True, argstr='-%s', position=-2, desc='Type of jacobian outcome')
    out_file = File(name_source=['trans_file'], name_template='%s', desc='The output jacobian determinant file name', argstr='%s', position=-1)