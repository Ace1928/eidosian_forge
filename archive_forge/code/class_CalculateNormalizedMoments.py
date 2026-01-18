import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class CalculateNormalizedMoments(BaseInterface):
    """
    Calculates moments of timeseries.

    Example
    -------
    >>> from nipype.algorithms import misc
    >>> skew = misc.CalculateNormalizedMoments()
    >>> skew.inputs.moment = 3
    >>> skew.inputs.timeseries_file = 'timeseries.txt'
    >>> skew.run() # doctest: +SKIP

    """
    input_spec = CalculateNormalizedMomentsInputSpec
    output_spec = CalculateNormalizedMomentsOutputSpec

    def _run_interface(self, runtime):
        self._moments = calc_moments(self.inputs.timeseries_file, self.inputs.moment)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['skewness'] = self._moments
        return outputs