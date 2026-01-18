import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class TractShredderOutputSpec(TraitedSpec):
    shredded = File(exists=True, desc='Shredded tract file')