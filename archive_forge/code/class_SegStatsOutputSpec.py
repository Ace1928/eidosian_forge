import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class SegStatsOutputSpec(TraitedSpec):
    summary_file = File(exists=True, desc='Segmentation summary statistics table')
    avgwf_txt_file = File(desc='Text file with functional statistics averaged over segs')
    avgwf_file = File(desc='Volume with functional statistics averaged over segs')
    sf_avg_file = File(desc='Text file with func statistics averaged over segs and framss')