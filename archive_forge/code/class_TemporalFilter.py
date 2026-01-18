import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class TemporalFilter(MathsCommand):
    """Use fslmaths to apply a low, high, or bandpass temporal filter to a
    timeseries.

    """
    input_spec = TemporalFilterInput
    _suffix = '_filt'