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
class RobustRegisterOutputSpec(TraitedSpec):
    out_reg_file = File(exists=True, desc='output registration file')
    registered_file = File(exists=True, desc='output image with registration applied')
    weights_file = File(exists=True, desc='image of weights used')
    half_source = File(exists=True, desc='source image mapped to halfway space')
    half_targ = File(exists=True, desc='target image mapped to halfway space')
    half_weights = File(exists=True, desc='weights image mapped to halfway space')
    half_source_xfm = File(exists=True, desc='transform file to map source image to halfway space')
    half_targ_xfm = File(exists=True, desc='transform file to map target image to halfway space')