import os
import numpy as np
from copy import deepcopy
from ase.calculators.calculator import KPoints, kpts2kpts
def _get_other(**params):
    out = []
    for kw, block in params.items():
        if kw in _special_kws:
            continue
        out += _format_block(kw, block)
    return out