import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
def _get_actual_outputroot(self, outputroot):
    actual_outputroot = os.path.join('procstream_outfiles', outputroot)
    return actual_outputroot