import warnings
from collections.abc import Iterable
from functools import wraps, cached_property
import ctypes
import numpy as np
from numpy.polynomial import Polynomial
from scipy._lib.doccer import (extend_notes_in_docstring,
from scipy._lib._ccallback import LowLevelCallable
from scipy import optimize
from scipy import integrate
import scipy.special as sc
import scipy.special._ufuncs as scu
from scipy._lib._util import _lazyselect, _lazywhere
from . import _stats
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
from ._distn_infrastructure import (
from ._ksstats import kolmogn, kolmognp, kolmogni
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
from ._censored_data import CensoredData
import scipy.stats._boost as _boost
from scipy.optimize import root_scalar
from scipy.stats._warnings_errors import FitError
import scipy.stats as stats
def _check_fit_input_parameters(dist, data, args, kwds):
    if not isinstance(data, CensoredData):
        data = np.asarray(data)
    floc = kwds.get('floc', None)
    fscale = kwds.get('fscale', None)
    num_shapes = len(dist.shapes.split(',')) if dist.shapes else 0
    fshape_keys = []
    fshapes = []
    if dist.shapes:
        shapes = dist.shapes.replace(',', ' ').split()
        for j, s in enumerate(shapes):
            key = 'f' + str(j)
            names = [key, 'f' + s, 'fix_' + s]
            val = _get_fixed_fit_value(kwds, names)
            fshape_keys.append(key)
            fshapes.append(val)
            if val is not None:
                kwds[key] = val
    known_keys = {'loc', 'scale', 'optimizer', 'method', 'floc', 'fscale', *fshape_keys}
    unknown_keys = set(kwds).difference(known_keys)
    if unknown_keys:
        raise TypeError(f'Unknown keyword arguments: {unknown_keys}.')
    if len(args) > num_shapes:
        raise TypeError('Too many positional arguments.')
    if None not in {floc, fscale, *fshapes}:
        raise RuntimeError('All parameters fixed. There is nothing to optimize.')
    uncensored = data._uncensor() if isinstance(data, CensoredData) else data
    if not np.isfinite(uncensored).all():
        raise ValueError('The data contains non-finite values.')
    return (data, *fshapes, floc, fscale)