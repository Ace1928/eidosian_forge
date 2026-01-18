import os
import warnings
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import load_json, save_json, split_filename
class CalcTopNCCInputSpec(CommandLineInputSpec):
    """Input Spec for CalcTopNCC."""
    in_file = File(argstr='-target %s', exists=True, mandatory=True, desc='Target file', position=1)
    num_templates = traits.Int(argstr='-templates %s', mandatory=True, position=2, desc='Number of Templates')
    in_templates = traits.List(File(exists=True), argstr='%s', position=3, mandatory=True)
    top_templates = traits.Int(argstr='-n %s', mandatory=True, position=4, desc='Number of Top Templates')
    mask_file = File(argstr='-mask %s', exists=True, desc='Filename of the ROI for label fusion')