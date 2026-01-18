from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
def _validate_sigs(self, typing_func, impl_func):
    typing_sig = utils.pysignature(typing_func)
    impl_sig = utils.pysignature(impl_func)

    def get_args_kwargs(sig):
        kws = []
        args = []
        pos_arg = None
        for x in sig.parameters.values():
            if x.default == utils.pyParameter.empty:
                args.append(x)
                if x.kind == utils.pyParameter.VAR_POSITIONAL:
                    pos_arg = x
                elif x.kind == utils.pyParameter.VAR_KEYWORD:
                    msg = "The use of VAR_KEYWORD (e.g. **kwargs) is unsupported. (offending argument name is '%s')"
                    raise InternalError(msg % x)
            else:
                kws.append(x)
        return (args, kws, pos_arg)
    ty_args, ty_kws, ty_pos = get_args_kwargs(typing_sig)
    im_args, im_kws, im_pos = get_args_kwargs(impl_sig)
    sig_fmt = 'Typing signature:         %s\nImplementation signature: %s'
    sig_str = sig_fmt % (typing_sig, impl_sig)
    err_prefix = 'Typing and implementation arguments differ in '
    a = ty_args
    b = im_args
    if ty_pos:
        if not im_pos:
            msg = "VAR_POSITIONAL (e.g. *args) argument kind (offending argument name is '%s') found in the typing function signature, but is not in the implementing function signature.\n%s" % (ty_pos, sig_str)
            raise InternalError(msg)
    elif im_pos:
        b = im_args[:im_args.index(im_pos)]
        try:
            a = ty_args[:ty_args.index(b[-1]) + 1]
        except ValueError:
            specialized = "argument names.\n%s\nFirst difference: '%s'"
            msg = err_prefix + specialized % (sig_str, b[-1])
            raise InternalError(msg)

    def gen_diff(typing, implementing):
        diff = set(typing) ^ set(implementing)
        return 'Difference: %s' % diff
    if a != b:
        specialized = 'argument names.\n%s\n%s' % (sig_str, gen_diff(a, b))
        raise InternalError(err_prefix + specialized)
    ty = [x.name for x in ty_kws]
    im = [x.name for x in im_kws]
    if ty != im:
        specialized = 'keyword argument names.\n%s\n%s'
        msg = err_prefix + specialized % (sig_str, gen_diff(ty_kws, im_kws))
        raise InternalError(msg)
    same = [x.default for x in ty_kws] == [x.default for x in im_kws]
    if not same:
        specialized = 'keyword argument default values.\n%s\n%s'
        msg = err_prefix + specialized % (sig_str, gen_diff(ty_kws, im_kws))
        raise InternalError(msg)