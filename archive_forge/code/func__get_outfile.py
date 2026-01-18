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
def _get_outfile(self):
    outfile = self.inputs.transformed_file
    if not isdefined(outfile):
        if self.inputs.inverse is True:
            if self.inputs.fs_target is True:
                src = 'orig.mgz'
            else:
                src = self.inputs.target_file
        else:
            src = self.inputs.source_file
        outfile = fname_presuffix(src, newpath=os.getcwd(), suffix='_warped')
    return outfile