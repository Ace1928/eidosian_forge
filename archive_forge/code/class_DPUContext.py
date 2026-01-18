import unittest
from numba.tests.support import TestCase
import ctypes
import operator
from functools import cached_property
import numpy as np
from numba import njit, types
from numba.extending import overload, intrinsic, overload_classmethod
from numba.core.target_extension import (
from numba.core import utils, fastmathpass, errors
from numba.core.dispatcher import Dispatcher
from numba.core.descriptors import TargetDescriptor
from numba.core import cpu, typing, cgutils
from numba.core.base import BaseContext
from numba.core.compiler_lock import global_compiler_lock
from numba.core import callconv
from numba.core.codegen import CPUCodegen, JITCodeLibrary
from numba.core.callwrapper import PyCallWrapper
from numba.core.imputils import RegistryLoader, Registry
from numba import _dynfunc
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.runtime import rtsys
from numba.core import compiler
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.typed_passes import PreLowerStripPhis
class DPUContext(BaseContext):
    allow_dynamic_globals = True

    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    @global_compiler_lock
    def init(self):
        self._internal_codegen = JITDPUCodegen('numba.exec')
        rtsys.initialize(self)
        self.refresh()

    def refresh(self):
        registry = dpu_function_registry
        try:
            loader = self._registries[registry]
        except KeyError:
            loader = RegistryLoader(registry)
            self._registries[registry] = loader
        self.install_registry(registry)
        self.typing_context.refresh()

    @property
    def target_data(self):
        return self._internal_codegen.target_data

    def codegen(self):
        return self._internal_codegen

    @cached_property
    def call_conv(self):
        return callconv.CPUCallConv(self)

    def get_env_body(self, builder, envptr):
        """
        From the given *envptr* (a pointer to a _dynfunc.Environment object),
        get a EnvBody allowing structured access to environment fields.
        """
        body_ptr = cgutils.pointer_add(builder, envptr, _dynfunc._impl_info['offsetof_env_body'])
        return cpu.EnvBody(self, builder, ref=body_ptr, cast_ref=True)

    def get_env_manager(self, builder):
        envgv = self.declare_env_global(builder.module, self.get_env_name(self.fndesc))
        envarg = builder.load(envgv)
        pyapi = self.get_python_api(builder)
        pyapi.emit_environment_sentry(envarg, debug_msg=self.fndesc.env_name)
        env_body = self.get_env_body(builder, envarg)
        return pyapi.get_env_manager(self.environment, env_body, envarg)

    def get_generator_state(self, builder, genptr, return_type):
        """
        From the given *genptr* (a pointer to a _dynfunc.Generator object),
        get a pointer to its state area.
        """
        return cgutils.pointer_add(builder, genptr, _dynfunc._impl_info['offsetof_generator_state'], return_type=return_type)

    def post_lowering(self, mod, library):
        if self.fastmath:
            fastmathpass.rewrite_module(mod, self.fastmath)
        library.add_linking_library(rtsys.library)

    def create_cpython_wrapper(self, library, fndesc, env, call_helper, release_gil=False):
        wrapper_module = self.create_module('wrapper')
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        wrapper_callee = llir.Function(wrapper_module, fnty, fndesc.llvm_func_name)
        builder = PyCallWrapper(self, wrapper_module, wrapper_callee, fndesc, env, call_helper=call_helper, release_gil=release_gil)
        builder.build()
        library.add_ir_module(wrapper_module)

    def create_cfunc_wrapper(self, library, fndesc, env, call_helper):
        pass

    def get_executable(self, library, fndesc, env):
        """
        Returns
        -------
        (cfunc, fnptr)

        - cfunc
            callable function (Can be None)
        - fnptr
            callable function address
        - env
            an execution environment (from _dynfunc)
        """
        fnptr = library.get_pointer_to_function(fndesc.llvm_cpython_wrapper_name)
        doc = 'compiled wrapper for %r' % (fndesc.qualname,)
        cfunc = _dynfunc.make_function(fndesc.lookup_module(), fndesc.qualname.split('.')[-1], doc, fnptr, env, (library,))
        library.codegen.set_env(self.get_env_name(fndesc), env)
        return cfunc