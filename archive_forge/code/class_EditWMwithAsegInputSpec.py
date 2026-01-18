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
class EditWMwithAsegInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', position=-4, mandatory=True, exists=True, desc='Input white matter segmentation file')
    brain_file = File(argstr='%s', position=-3, mandatory=True, exists=True, desc='Input brain/T1 file')
    seg_file = File(argstr='%s', position=-2, mandatory=True, exists=True, desc='Input presurf segmentation file')
    out_file = File(argstr='%s', position=-1, mandatory=True, exists=False, desc='File to be written as output')
    keep_in = traits.Bool(argstr='-keep-in', desc='Keep edits as found in input volume')