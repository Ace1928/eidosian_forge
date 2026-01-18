import re
from functools import cached_property
import llvmlite.binding as ll
from llvmlite import ir
from numba.core import (cgutils, config, debuginfo, itanium_mangler, types,
from numba.core.dispatcher import Dispatcher
from numba.core.base import BaseContext
from numba.core.callconv import BaseCallConv, MinimalCallConv
from numba.core.typing import cmathdecl
from numba.core import datamodel
from .cudadrv import nvvm
from numba.cuda import codegen, nvvmutils, ufuncs
from numba.cuda.models import cuda_data_manager
class CUDATypingContext(typing.BaseContext):

    def load_additional_registries(self):
        from . import cudadecl, cudamath, libdevicedecl, vector_types
        from numba.core.typing import enumdecl, cffi_utils
        self.install_registry(cudadecl.registry)
        self.install_registry(cffi_utils.registry)
        self.install_registry(cudamath.registry)
        self.install_registry(cmathdecl.registry)
        self.install_registry(libdevicedecl.registry)
        self.install_registry(enumdecl.registry)
        self.install_registry(vector_types.typing_registry)

    def resolve_value_type(self, val):
        from numba.cuda.dispatcher import CUDADispatcher
        if isinstance(val, Dispatcher) and (not isinstance(val, CUDADispatcher)):
            try:
                val = val.__dispatcher
            except AttributeError:
                if not val._can_compile:
                    raise ValueError('using cpu function on device but its compilation is disabled')
                targetoptions = val.targetoptions.copy()
                targetoptions['device'] = True
                targetoptions['debug'] = targetoptions.get('debug', False)
                targetoptions['opt'] = targetoptions.get('opt', True)
                disp = CUDADispatcher(val.py_func, targetoptions)
                val.__dispatcher = disp
                val = disp
        return super(CUDATypingContext, self).resolve_value_type(val)