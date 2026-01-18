import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class HistOutputSpec(TraitedSpec):
    out_file = File(desc='output file', exists=True)
    out_show = File(desc='output visual histogram')