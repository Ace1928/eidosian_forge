import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
class CallableTypeRef(types.Callable):

    def __init__(self, instance_type):
        self.instance_type = instance_type
        self.sig_to_impl_key = {}
        self.compiled_templates = []
        super(CallableTypeRef, self).__init__('callable_type_ref[{}]'.format(self.instance_type))

    def get_call_type(self, context, args, kws):
        res_sig = None
        for template in context._functions[type(self)]:
            try:
                res_sig = template.apply(args, kws)
            except Exception:
                pass
            else:
                compiled_ovlds = getattr(template, '_compiled_overloads', {})
                if args in compiled_ovlds:
                    self.sig_to_impl_key[res_sig] = compiled_ovlds[args]
                    self.compiled_templates.append(template)
                    break
        return res_sig

    def get_call_signatures(self):
        sigs = list(self.sig_to_impl_key.keys())
        return (sigs, True)

    def get_impl_key(self, sig):
        return self.sig_to_impl_key[sig]