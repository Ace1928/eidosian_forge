import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class UnifizeOutputSpec(TraitedSpec):
    scale_file = File(desc='scale factor file')
    out_file = File(desc='unifized file', exists=True)