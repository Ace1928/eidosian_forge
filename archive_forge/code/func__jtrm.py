from __future__ import division, print_function, absolute_import
import math
import numpy as np
from ..util import import_
from ..core import RecoverableError
from ..symbolic import ScaledSys
def _jtrm(ri, ji):
    if logc and ri == ji:
        return -f[ri] / liny[ri]
    else:
        return 0