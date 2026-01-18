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
class ComputeDVARSInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='functional data, after HMC')
    in_mask = File(exists=True, mandatory=True, desc='a brain mask')
    remove_zerovariance = traits.Bool(True, usedefault=True, desc='remove voxels with zero variance')
    variance_tol = traits.Float(1e-07, usedefault=True, desc='maximum variance to consider "close to" zero for the purposes of removal')
    save_std = traits.Bool(True, usedefault=True, desc='save standardized DVARS')
    save_nstd = traits.Bool(False, usedefault=True, desc='save non-standardized DVARS')
    save_vxstd = traits.Bool(False, usedefault=True, desc='save voxel-wise standardized DVARS')
    save_all = traits.Bool(False, usedefault=True, desc='output all DVARS')
    series_tr = traits.Float(desc='repetition time in sec.')
    save_plot = traits.Bool(False, usedefault=True, desc='write DVARS plot')
    figdpi = traits.Int(100, usedefault=True, desc='output dpi for the plot')
    figsize = traits.Tuple(traits.Float(11.7), traits.Float(2.3), usedefault=True, desc='output figure size')
    figformat = traits.Enum('png', 'pdf', 'svg', usedefault=True, desc='output format for figures')
    intensity_normalization = traits.Float(1000.0, usedefault=True, desc='Divide value in each voxel at each timepoint by the median calculated across all voxelsand timepoints within the mask (if specified)and then multiply by the value specified bythis parameter. By using the default (1000)output DVARS will be expressed in x10 % BOLD units compatible with Power et al.2012. Set this to 0 to disable intensitynormalization altogether.')