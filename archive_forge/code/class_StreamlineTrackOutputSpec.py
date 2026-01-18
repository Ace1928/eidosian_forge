import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class StreamlineTrackOutputSpec(TraitedSpec):
    tracked = File(exists=True, desc='output file containing reconstructed tracts')