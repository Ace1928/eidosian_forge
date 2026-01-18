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
class SegmentWMInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', exists=True, mandatory=True, position=-2, desc='Input file for SegmentWM')
    out_file = File(argstr='%s', exists=False, mandatory=True, position=-1, desc='File to be written as output for SegmentWM')