import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class NormOutputSpec(TraitedSpec):
    output_file = File(desc='output file', exists=True)
    output_threshold_mask = File(desc='threshold mask file')