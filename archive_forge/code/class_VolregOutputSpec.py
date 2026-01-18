import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class VolregOutputSpec(TraitedSpec):
    out_file = File(desc='registered file', exists=True)
    md1d_file = File(desc='max displacement info file', exists=True)
    oned_file = File(desc='movement parameters info file', exists=True)
    oned_matrix_save = File(desc='matrix transformation from base to input', exists=True)