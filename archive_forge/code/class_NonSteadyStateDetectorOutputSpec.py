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
class NonSteadyStateDetectorOutputSpec(TraitedSpec):
    n_volumes_to_discard = traits.Int(desc='Number of non-steady state volumesdetected in the beginning of the scan.')