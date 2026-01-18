import os
import warnings
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import load_json, save_json, split_filename
class LabelFusionOutput(TraitedSpec):
    """Output Spec for LabelFusion."""
    out_file = File(exists=True, desc='image written after calculations')