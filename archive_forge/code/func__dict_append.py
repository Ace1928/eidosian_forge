import sys
from distutils.core import Distribution
import warnings
import distutils.core
import distutils.dist
from numpy.distutils.extension import Extension  # noqa: F401
from numpy.distutils.numpy_distribution import NumpyDistribution
from numpy.distutils.command import config, config_compiler, \
from numpy.distutils.misc_util import is_sequence, is_string
def _dict_append(d, **kws):
    for k, v in kws.items():
        if k not in d:
            d[k] = v
            continue
        dv = d[k]
        if isinstance(dv, tuple):
            d[k] = dv + tuple(v)
        elif isinstance(dv, list):
            d[k] = dv + list(v)
        elif isinstance(dv, dict):
            _dict_append(dv, **v)
        elif is_string(dv):
            assert is_string(v)
            d[k] = v
        else:
            raise TypeError(repr(type(dv)))