import platform
import llvmlite.binding as ll
from llvmlite import ir
from numba import _dynfunc
from numba.core.callwrapper import PyCallWrapper
from numba.core.base import BaseContext
from numba.core import (utils, types, config, cgutils, callconv, codegen,
from numba.core.options import TargetOptions, include_default_options
from numba.core.runtime import rtsys
from numba.core.compiler_lock import global_compiler_lock
import numba.core.entrypoints
from numba.core.cpu_options import (ParallelOptions, # noqa F401
from numba.np import ufunc_db
class CPUTargetOptions(_options_mixin, TargetOptions):

    def finalize(self, flags, options):
        if not flags.is_set('enable_pyobject'):
            flags.enable_pyobject = True
        if not flags.is_set('enable_looplift'):
            flags.enable_looplift = True
        flags.inherit_if_not_set('nrt', default=True)
        if not flags.is_set('debuginfo'):
            flags.debuginfo = config.DEBUGINFO_DEFAULT
        if not flags.is_set('dbg_extend_lifetimes'):
            if flags.debuginfo:
                flags.dbg_extend_lifetimes = True
            else:
                flags.dbg_extend_lifetimes = config.EXTEND_VARIABLE_LIFETIMES
        if not flags.is_set('boundscheck'):
            flags.boundscheck = flags.debuginfo
        flags.enable_pyobject_looplift = True
        flags.inherit_if_not_set('fastmath')
        flags.inherit_if_not_set('error_model', default='python')
        flags.inherit_if_not_set('target_backend')
        flags.inherit_if_not_set('forceinline')
        if flags.forceinline:
            flags.optnone = False