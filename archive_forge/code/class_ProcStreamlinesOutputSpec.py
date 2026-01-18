import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class ProcStreamlinesOutputSpec(TraitedSpec):
    proc = File(exists=True, desc='Processed Streamlines')
    outputroot_files = OutputMultiPath(File(exists=True))