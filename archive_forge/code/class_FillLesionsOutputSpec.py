import warnings
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class FillLesionsOutputSpec(TraitedSpec):
    """Output Spec for FillLesions."""
    out_file = File(desc='Output segmentation')