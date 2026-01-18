import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FindTheBiggestOutputSpec(TraitedSpec):
    out_file = File(exists=True, argstr='%s', desc='output file indexed in order of input files')