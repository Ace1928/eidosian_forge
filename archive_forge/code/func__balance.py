import copy
from collections import OrderedDict
from math import log2
import numpy as np
from .. import functions as fn
def _balance(self):
    iso = self.iso
    light = self.lightMeter
    sh = self.shutter
    ap = self.aperture
    bal = 4.0 / ap * (sh / (1.0 / 60.0)) * (iso / 100.0) * 2 ** light
    return log2(bal)