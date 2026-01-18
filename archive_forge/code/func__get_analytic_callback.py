from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
@staticmethod
def _get_analytic_callback(ori_sys, analytic_exprs, new_dep, new_params):
    return _Callback(ori_sys.indep, new_dep, new_params, analytic_exprs, Lambdify=ori_sys.be.Lambdify)