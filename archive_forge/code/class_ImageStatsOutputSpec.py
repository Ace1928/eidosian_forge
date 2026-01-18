import os
from ..base import (
from ...utils.filemanip import split_filename
class ImageStatsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Path of the file computed with the statistic chosen')