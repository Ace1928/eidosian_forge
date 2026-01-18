from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class DiffeoScalarVolOutputSpec(TraitedSpec):
    out_file = File(desc='moved volume', exists=True)