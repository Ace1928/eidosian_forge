import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BBoxOutputSpec(TraitedSpec):
    output_file = File(desc='output file containing bounding box corners', exists=True)