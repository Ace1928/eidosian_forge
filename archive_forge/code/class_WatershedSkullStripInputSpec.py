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
class WatershedSkullStripInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', exists=True, mandatory=True, position=-2, desc='input volume')
    out_file = File('brainmask.auto.mgz', argstr='%s', exists=False, mandatory=True, position=-1, usedefault=True, desc='output volume')
    t1 = traits.Bool(argstr='-T1', desc='specify T1 input volume (T1 grey value = 110)')
    brain_atlas = File(argstr='-brain_atlas %s', exists=True, position=-4, desc='')
    transform = File(argstr='%s', exists=False, position=-3, desc='undocumented')