import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class Tracks2ProbOutputSpec(TraitedSpec):
    tract_image = File(exists=True, desc='Output tract count or track density image')