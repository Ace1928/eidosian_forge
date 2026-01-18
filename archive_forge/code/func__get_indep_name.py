from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _get_indep_name(names):
    if 'x' not in names:
        indep_name = 'x'
    else:
        i = 0
        indep_name = 'indep0'
        while indep_name in names:
            i += 1
            indep_name = 'indep%d' % i
    return indep_name