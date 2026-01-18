from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
def _get_kernel_source(self, template, sig, funcname):
    args = ['a%d' % i for i in range(len(sig.args))]
    fmts = dict(name=funcname, args=', '.join(args), argitems=', '.join(('%s[__tid__]' % i for i in args)))
    return template.format(**fmts)