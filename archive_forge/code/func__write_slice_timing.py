import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
def _write_slice_timing(self):
    slice_timing = list(self.inputs.slice_timing)
    if self.inputs.slice_encoding_direction.endswith('-'):
        slice_timing.reverse()
    fname = 'slice_timing.1D'
    with open(fname, 'w') as fobj:
        fobj.write('\t'.join(map(str, slice_timing)))
    return fname