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
class ReconAllOutputSpec(FreeSurferSource.output_spec):
    subjects_dir = Directory(exists=True, desc='Freesurfer subjects directory.')
    subject_id = traits.Str(desc='Subject name for whom to retrieve data')