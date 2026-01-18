import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class OutlierCountOutputSpec(TraitedSpec):
    out_outliers = File(exists=True, desc='output image file name')
    out_file = File(desc='capture standard output')