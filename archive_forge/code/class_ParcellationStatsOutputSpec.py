import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ParcellationStatsOutputSpec(TraitedSpec):
    out_table = File(exists=False, desc='Table output to tablefile')
    out_color = File(exists=False, desc="Output annotation files's colortable to text file")