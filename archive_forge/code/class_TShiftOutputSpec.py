import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TShiftOutputSpec(AFNICommandOutputSpec):
    timing_file = File(desc='AFNI formatted timing file, if ``slice_timing`` is a list')