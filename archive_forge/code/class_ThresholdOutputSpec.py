import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class ThresholdOutputSpec(TraitedSpec):
    thresholded_map = File(exists=True)
    n_clusters = traits.Int()
    pre_topo_fdr_map = File(exists=True)
    pre_topo_n_clusters = traits.Int()
    activation_forced = traits.Bool()
    cluster_forming_thr = traits.Float()