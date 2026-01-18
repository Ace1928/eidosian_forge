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
def _get_expert_file(self):
    if isdefined(self.inputs.subjects_dir):
        subjects_dir = self.inputs.subjects_dir
    else:
        subjects_dir = self._gen_subjects_dir()
    xopts_file = os.path.join(subjects_dir, self.inputs.subject_id, 'scripts', 'expert-options')
    if not os.path.exists(xopts_file):
        return ''
    with open(xopts_file, 'r') as fobj:
        return fobj.read()