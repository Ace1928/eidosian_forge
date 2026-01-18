import copy
from collections import OrderedDict
from math import log2
import numpy as np
from .. import functions as fn
def _aperture(self):
    """
            Determine aperture automatically under a variety of conditions.
            """
    iso = self.iso
    exp = self.exposure
    light = self.lightMeter
    try:
        sh = self.shutter
        ap = 4.0 * (sh / (1.0 / 60.0)) * (iso / 100.0) * 2 ** exp * 2 ** light
        ap = fn.clip_scalar(ap, 2.0, 16.0)
    except RuntimeError:
        sh = 1.0 / 60.0
        raise
    return ap