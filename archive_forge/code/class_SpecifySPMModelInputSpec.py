from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
class SpecifySPMModelInputSpec(SpecifyModelInputSpec):
    concatenate_runs = traits.Bool(False, usedefault=True, desc='Concatenate all runs to look like a single session.')
    output_units = traits.Enum('secs', 'scans', usedefault=True, desc='Units of design event onsets and durations (secs or scans)')