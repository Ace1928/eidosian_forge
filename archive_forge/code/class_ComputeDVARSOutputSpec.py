import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
class ComputeDVARSOutputSpec(TraitedSpec):
    out_std = File(exists=True, desc='output text file')
    out_nstd = File(exists=True, desc='output text file')
    out_vxstd = File(exists=True, desc='output text file')
    out_all = File(exists=True, desc='output text file')
    avg_std = traits.Float()
    avg_nstd = traits.Float()
    avg_vxstd = traits.Float()
    fig_std = File(exists=True, desc='output DVARS plot')
    fig_nstd = File(exists=True, desc='output DVARS plot')
    fig_vxstd = File(exists=True, desc='output DVARS plot')