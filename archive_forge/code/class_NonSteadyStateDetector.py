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
class NonSteadyStateDetector(BaseInterface):
    """
    Returns the number of non-steady state volumes detected at the beginning
    of the scan.
    """
    input_spec = NonSteadyStateDetectorInputSpec
    output_spec = NonSteadyStateDetectorOutputSpec

    def _run_interface(self, runtime):
        in_nii = nb.load(self.inputs.in_file)
        global_signal = in_nii.dataobj[:, :, :, :50].mean(axis=0).mean(axis=0).mean(axis=0)
        self._results = {'n_volumes_to_discard': is_outlier(global_signal)}
        return runtime

    def _list_outputs(self):
        return self._results