import os
import numpy as np
from copy import deepcopy
from ase.calculators.calculator import KPoints, kpts2kpts
def _get_set(**params):
    return ['set ' + _format_line(key, val) for key, val in params.items()]