from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def _get_cneqsys3(small):
    return ConditionalNeqSys([(lambda x, p: x[0] * x[1] > p[3], lambda x, p: x[2] > math.exp(small))], _factory_log(small))