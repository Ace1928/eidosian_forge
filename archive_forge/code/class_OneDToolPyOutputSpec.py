import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class OneDToolPyOutputSpec(AFNICommandOutputSpec):
    out_file = File(desc='output of 1D_tool.py')