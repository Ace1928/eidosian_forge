import warnings
import numpy as np
from . import cysoxr
from .cysoxr import QQ, LQ, MQ, HQ, VHQ
from ._version import version as __version__
def _resample_oneshot(x, in_rate: float, out_rate: float, quality='HQ'):
    """
    Resample using libsoxr's `soxr_oneshot()`. Use `resample()` for general use.
    `soxr_oneshot()` becomes slow with long input.
    This function exists for test purpose.
    """
    return cysoxr.cysoxr_oneshot(in_rate, out_rate, x, _quality_to_enum(quality))