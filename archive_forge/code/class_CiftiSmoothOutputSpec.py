from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import WBCommand
from ... import logging
class CiftiSmoothOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output CIFTI file')