from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def _factory_lin(conds):
    precip = conds[0]

    def cb(x, p):
        f = [None] * 3
        f[0] = x[0] + x[2] - p[0] - p[2]
        f[1] = x[1] + x[2] - p[1] - p[2]
        if precip:
            f[2] = x[0] * x[1] - p[3]
        else:
            f[2] = x[2]
        return f
    return NeqSys(3, 3, cb)