import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegAverageOutputSpec(TraitedSpec):
    """Output Spec for RegAverage."""
    out_file = File(desc='Output file name')