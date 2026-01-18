import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class AutomaskOutputSpec(TraitedSpec):
    out_file = File(desc='mask file', exists=True)
    brain_file = File(desc='brain file (skull stripped)', exists=True)