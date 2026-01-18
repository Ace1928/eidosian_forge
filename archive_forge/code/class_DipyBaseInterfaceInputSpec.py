import os.path as op
import inspect
import numpy as np
from ... import logging
from ..base import (
class DipyBaseInterfaceInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input diffusion data')
    in_bval = File(exists=True, mandatory=True, desc='input b-values table')
    in_bvec = File(exists=True, mandatory=True, desc='input b-vectors table')
    b0_thres = traits.Int(700, usedefault=True, desc='b0 threshold')
    out_prefix = traits.Str(desc='output prefix for file names')