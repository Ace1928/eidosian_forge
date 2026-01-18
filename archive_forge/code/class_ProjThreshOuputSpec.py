import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ProjThreshOuputSpec(TraitedSpec):
    out_files = traits.List(File(exists=True), desc='path/name of output volume after thresholding')