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
class DeviceGUFuncVectorize(_BaseUFuncBuilder):

    def __init__(self, func, sig, identity=None, cache=False, targetoptions={}, writable_args=()):
        if cache:
            raise TypeError('caching is not supported')
        if writable_args:
            raise TypeError('writable_args are not supported')
        if not targetoptions.pop('nopython', True):
            raise TypeError('nopython flag must be True')
        if targetoptions:
            opts = ', '.join([repr(k) for k in targetoptions.keys()])
            fmt = 'The following target options are not supported: {0}'
            raise TypeError(fmt.format(opts))
        self.py_func = func
        self.identity = parse_identity(identity)
        self.signature = sig
        self.inputsig, self.outputsig = parse_signature(self.signature)
        self.kernelmap = OrderedDict()

    @property
    def pyfunc(self):
        return self.py_func

    def add(self, sig=None):
        indims = [len(x) for x in self.inputsig]
        outdims = [len(x) for x in self.outputsig]
        args, return_type = sigutils.normalize_signature(sig)
        valid_return_type = return_type in (types.none, None)
        if not valid_return_type:
            raise TypeError(f'guvectorized functions cannot return values: signature {sig} specifies {return_type} return type')
        funcname = self.py_func.__name__
        src = expand_gufunc_template(self._kernel_template, indims, outdims, funcname, args)
        glbls = self._get_globals(sig)
        exec(src, glbls)
        fnobj = glbls['__gufunc_{name}'.format(name=funcname)]
        outertys = list(_determine_gufunc_outer_types(args, indims + outdims))
        kernel = self._compile_kernel(fnobj, sig=tuple(outertys))
        nout = len(outdims)
        dtypes = [np.dtype(str(t.dtype)) for t in outertys]
        indtypes = tuple(dtypes[:-nout])
        outdtypes = tuple(dtypes[-nout:])
        self.kernelmap[indtypes] = (outdtypes, kernel)

    def _compile_kernel(self, fnobj, sig):
        raise NotImplementedError

    def _get_globals(self, sig):
        raise NotImplementedError