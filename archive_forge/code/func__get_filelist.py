import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
def _get_filelist(self, outdir):
    """Returns list of files to be converted"""
    filemap = {}
    for f in self._get_dicomfiles():
        head, fname = os.path.split(f)
        fname, ext = os.path.splitext(fname)
        fileparts = fname.split('-')
        runno = int(fileparts[1])
        out_type = MRIConvert.filemap[self.inputs.out_type]
        outfile = os.path.join(outdir, '.'.join(('%s-%02d' % (fileparts[0], runno), out_type)))
        filemap[runno] = (f, outfile)
    if self.inputs.dicom_info:
        files = [filemap[r] for r in self._get_runs()]
    else:
        files = [filemap[r] for r in list(filemap.keys())]
    return files