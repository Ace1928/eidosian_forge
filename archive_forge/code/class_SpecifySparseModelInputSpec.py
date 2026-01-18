from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
class SpecifySparseModelInputSpec(SpecifyModelInputSpec):
    time_acquisition = traits.Float(0, mandatory=True, desc='Time in seconds to acquire a single image volume')
    volumes_in_cluster = traits.Range(1, usedefault=True, desc='Number of scan volumes in a cluster')
    model_hrf = traits.Bool(desc='Model sparse events with hrf')
    stimuli_as_impulses = traits.Bool(True, desc='Treat each stimulus to be impulse-like', usedefault=True)
    use_temporal_deriv = traits.Bool(requires=['model_hrf'], desc='Create a temporal derivative in addition to regular regressor')
    scale_regressors = traits.Bool(True, desc='Scale regressors by the peak', usedefault=True)
    scan_onset = traits.Float(0.0, desc='Start of scanning relative to onset of run in secs', usedefault=True)
    save_plot = traits.Bool(desc='Save plot of sparse design calculation (requires matplotlib)')