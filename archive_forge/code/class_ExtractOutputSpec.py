import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class ExtractOutputSpec(TraitedSpec):
    output_file = File(desc='output file in raw/text format', exists=True)