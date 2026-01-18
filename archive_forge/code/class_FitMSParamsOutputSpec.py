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
class FitMSParamsOutputSpec(TraitedSpec):
    t1_image = File(exists=True, desc='image of estimated T1 relaxation values')
    pd_image = File(exists=True, desc='image of estimated proton density values')
    t2star_image = File(exists=True, desc='image of estimated T2* values')