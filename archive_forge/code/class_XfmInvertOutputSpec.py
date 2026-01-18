import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class XfmInvertOutputSpec(TraitedSpec):
    output_file = File(desc='output file', exists=True)
    output_grid = File(desc='output grid file', exists=True)