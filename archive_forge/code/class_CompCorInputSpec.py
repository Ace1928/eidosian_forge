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
class CompCorInputSpec(BaseInterfaceInputSpec):
    realigned_file = File(exists=True, mandatory=True, desc='already realigned brain image (4D)')
    mask_files = InputMultiPath(File(exists=True), desc='One or more mask files that determines ROI (3D). When more that one file is provided ``merge_method`` or ``merge_index`` must be provided')
    merge_method = traits.Enum('union', 'intersect', 'none', xor=['mask_index'], requires=['mask_files'], desc='Merge method if multiple masks are present - ``union`` uses voxels included in at least one input mask, ``intersect`` uses only voxels present in all input masks, ``none`` performs CompCor on each mask individually')
    mask_index = traits.Range(low=0, xor=['merge_method'], requires=['mask_files'], desc='Position of mask in ``mask_files`` to use - first is the default.')
    mask_names = traits.List(traits.Str, desc='Names for provided masks (for printing into metadata). If provided, it must be as long as the final mask list (after any merge and indexing operations).')
    components_file = traits.Str('components_file.txt', usedefault=True, desc='Filename to store physiological components')
    num_components = traits.Either('all', traits.Range(low=1), xor=['variance_threshold'], desc='Number of components to return from the decomposition. If ``num_components`` is ``all``, then all components will be retained.')
    variance_threshold = traits.Range(low=0.0, high=1.0, exclude_low=True, exclude_high=True, xor=['num_components'], desc='Select the number of components to be returned automatically based on their ability to explain variance in the dataset. ``variance_threshold`` is a fractional value between 0 and 1; the number of components retained will be equal to the minimum number of components necessary to explain the provided fraction of variance in the masked time series.')
    pre_filter = traits.Enum('polynomial', 'cosine', False, usedefault=True, desc='Detrend time series prior to component extraction')
    use_regress_poly = traits.Bool(deprecated='0.15.0', new_name='pre_filter', desc='use polynomial regression pre-component extraction')
    regress_poly_degree = traits.Range(low=1, value=1, usedefault=True, desc='the degree polynomial to use')
    header_prefix = traits.Str(desc='the desired header for the output tsv file (one column). If undefined, will default to "CompCor"')
    high_pass_cutoff = traits.Float(128, usedefault=True, desc='Cutoff (in seconds) for "cosine" pre-filter')
    repetition_time = traits.Float(desc='Repetition time (TR) of series - derived from image header if unspecified')
    save_pre_filter = traits.Either(traits.Bool, File, default=False, usedefault=True, desc='Save pre-filter basis as text file')
    save_metadata = traits.Either(traits.Bool, File, default=False, usedefault=True, desc='Save component metadata as text file')
    ignore_initial_volumes = traits.Range(low=0, usedefault=True, desc='Number of volumes at start of series to ignore')
    failure_mode = traits.Enum('error', 'NaN', usedefault=True, desc='When no components are found or convergence fails, raise an error or silently return columns of NaNs.')