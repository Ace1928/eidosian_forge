import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class AnalyzeHeaderOutputSpec(TraitedSpec):
    header = File(exists=True, desc='Analyze header')