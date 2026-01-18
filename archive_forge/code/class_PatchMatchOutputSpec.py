import warnings
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class PatchMatchOutputSpec(TraitedSpec):
    """OutputSpec for PatchMatch."""
    out_file = File(desc='Output segmentation')