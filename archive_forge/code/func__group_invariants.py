from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _group_invariants(all_invar, deps, be, names=None):
    linear_invar = []
    nonlinear_invar = []
    lin_names, nonlin_names = ([], [])
    use_names = names is not None and len(names) > 0
    for idx, invar in enumerate(all_invar):
        derivs = [invar.diff(dep) for dep in deps]
        if all([deriv.is_Number for deriv in derivs]):
            linear_invar.append(derivs)
            if use_names:
                lin_names.append(names[idx])
        else:
            nonlinear_invar.append(invar)
            if use_names:
                nonlin_names.append(names[idx])
    if names is None:
        return (linear_invar, nonlinear_invar)
    else:
        return (linear_invar, nonlinear_invar, lin_names, nonlin_names)