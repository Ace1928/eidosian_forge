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
class FitMSParamsInputSpec(FSTraitedSpec):
    in_files = traits.List(File(exists=True), argstr='%s', position=-2, mandatory=True, desc='list of FLASH images (must be in mgh format)')
    tr_list = traits.List(traits.Int, desc='list of TRs of the input files (in msec)')
    te_list = traits.List(traits.Float, desc='list of TEs of the input files (in msec)')
    flip_list = traits.List(traits.Int, desc='list of flip angles of the input files')
    xfm_list = traits.List(File(exists=True), desc='list of transform files to apply to each FLASH image')
    out_dir = Directory(argstr='%s', position=-1, genfile=True, desc='directory to store output in')