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
class NormalizeInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', exists=True, mandatory=True, position=-2, desc='The input file for Normalize')
    out_file = File(argstr='%s', position=-1, name_source=['in_file'], name_template='%s_norm', hash_files=False, keep_extension=True, desc='The output file for Normalize')
    gradient = traits.Int(argstr='-g %d', desc='use max intensity/mm gradient g (default=1)')
    mask = File(argstr='-mask %s', exists=True, desc='The input mask file for Normalize')
    segmentation = File(argstr='-aseg %s', exists=True, desc='The input segmentation for Normalize')
    transform = File(exists=True, desc='Transform file from the header of the input file')