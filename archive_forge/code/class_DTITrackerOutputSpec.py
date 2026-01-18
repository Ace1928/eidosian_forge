import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
class DTITrackerOutputSpec(TraitedSpec):
    track_file = File(exists=True)
    mask_file = File(exists=True)