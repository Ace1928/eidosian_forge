import os
import warnings
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import load_json, save_json, split_filename
class CalcTopNCCOutputSpec(TraitedSpec):
    """Output Spec for CalcTopNCC."""
    out_files = traits.Any(File(exists=True))