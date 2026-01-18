from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _get_ny_nparams_from_kw(ny, nparams, kwargs):
    if kwargs.get('dep_by_name', False):
        if 'names' not in kwargs:
            raise ValueError('Need ``names`` in kwargs.')
        if ny is None:
            ny = len(kwargs['names'])
        elif ny != len(kwargs['names']):
            raise ValueError('Inconsistent between ``ny`` and length of ``names``.')
    if kwargs.get('par_by_name', False):
        if 'param_names' not in kwargs:
            raise ValueError('Need ``param_names`` in kwargs.')
        if nparams is None:
            nparams = len(kwargs['param_names'])
        elif nparams != len(kwargs['param_names']):
            raise ValueError('Inconsistent between ``nparams`` and length of ``param_names``.')
    if nparams is None:
        nparams = 0
    if ny is None:
        raise ValueError('Need ``ny`` or ``names`` together with ``dep_by_name==True``.')
    if kwargs.get('names', None) is not None and kwargs.get('param_names', None) is not None:
        all_names = set.union(set(kwargs['names']), set(kwargs['param_names']))
        if len(all_names) < len(kwargs['names']) + len(kwargs['param_names']):
            raise ValueError('Names of dependent variables cannot be used a parameter names')
    return (ny, nparams)