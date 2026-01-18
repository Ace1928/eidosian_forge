import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
def _trk_to_coords(self, in_file, out_file=None):
    from nibabel.trackvis import TrackvisFile
    trkfile = TrackvisFile.from_file(in_file)
    streamlines = trkfile.streamlines
    if out_file is None:
        out_file, _ = op.splitext(in_file)
    np.savetxt(out_file + '.txt', streamlines)
    return out_file + '.txt'