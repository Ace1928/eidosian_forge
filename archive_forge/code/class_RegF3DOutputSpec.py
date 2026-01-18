import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegF3DOutputSpec(TraitedSpec):
    """Output Spec for RegF3D."""
    cpp_file = File(desc='The output CPP file')
    res_file = File(desc='The output resampled image')
    invcpp_file = File(desc='The output inverse CPP file')
    invres_file = File(desc='The output inverse res file')
    desc = 'Output string in the format for reg_average'
    avg_output = traits.String(desc=desc)