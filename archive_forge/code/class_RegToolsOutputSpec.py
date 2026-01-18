import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegToolsOutputSpec(TraitedSpec):
    """Output Spec for RegTools."""
    out_file = File(desc='The output file', exists=True)