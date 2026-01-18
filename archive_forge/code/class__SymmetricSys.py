from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
class _SymmetricSys(SuperClass):

    def __init__(self, dep_exprs, indep=None, **inner_kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs.update(inner_kwargs)
        dep, exprs = zip(*dep_exprs)
        super(_SymmetricSys, self).__init__(zip(dep, exprs), indep, dep_transf=list(zip(list(map(dep_tr[0], dep)), list(map(dep_tr[1], dep)))) if dep_tr is not None else None, indep_transf=(indep_tr[0](indep), indep_tr[1](indep)) if indep_tr is not None else None, **new_kwargs)

    @classmethod
    def from_callback(cls, cb, ny=None, nparams=None, **inner_kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs.update(inner_kwargs)
        return SuperClass.from_callback(cb, ny, nparams, dep_transf_cbs=repeat(dep_tr) if dep_tr is not None else None, indep_transf_cbs=indep_tr, **new_kwargs)