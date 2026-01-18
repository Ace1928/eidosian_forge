import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class GCORInputSpec(CommandLineInputSpec):
    in_file = File(desc='input dataset to compute the GCOR over', argstr='-input %s', position=-1, mandatory=True, exists=True, copyfile=False)
    mask = File(desc='mask dataset, for restricting the computation', argstr='-mask %s', exists=True, copyfile=False)
    nfirst = traits.Int(0, argstr='-nfirst %d', desc='specify number of initial TRs to ignore')
    no_demean = traits.Bool(False, argstr='-no_demean', desc='do not (need to) demean as first step')