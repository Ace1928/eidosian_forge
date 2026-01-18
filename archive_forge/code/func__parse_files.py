import os
import re
from copy import deepcopy
import itertools as it
import glob
from glob import iglob
from ..utils.filemanip import split_filename
from .base import (
def _parse_files(self, filenames):
    outfiles, bvals, bvecs, bids = ([], [], [], [])
    outtypes = ['.bval', '.bvec', '.json', '.txt']
    if self.inputs.to_nrrd:
        outtypes += ['.nrrd', '.nhdr', '.raw.gz']
    else:
        outtypes += ['.nii', '.nii.gz']
    for filename in filenames:
        for fl in search_files(filename, outtypes):
            if fl.endswith('.nii') or fl.endswith('.gz') or fl.endswith('.nrrd') or fl.endswith('.nhdr'):
                outfiles.append(fl)
            elif fl.endswith('.bval'):
                bvals.append(fl)
            elif fl.endswith('.bvec'):
                bvecs.append(fl)
            elif fl.endswith('.json') or fl.endswith('.txt'):
                bids.append(fl)
    self.output_files = outfiles
    self.bvecs = bvecs
    self.bvals = bvals
    self.bids = bids