import warnings
import functools
import locale
import weakref
import ctypes
import html
import textwrap
import llvmlite.binding as ll
import llvmlite.ir as llvmir
from abc import abstractmethod, ABCMeta
from numba.core import utils, config, cgutils
from numba.core.llvm_bindings import create_pass_manager_builder
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct
from numba.core.runtime import rtsys
from numba.core.compiler_lock import require_global_compiler_lock
from numba.core.errors import NumbaInvalidConfigWarning
from numba.misc.inspection import disassemble_elf_to_cfg
from numba.misc.llvm_pass_timings import PassTimingsCollection
@classmethod
def _unserialize(cls, codegen, state):
    name, kind, data = state
    self = codegen.create_library(name)
    assert isinstance(self, cls)
    if kind == 'bitcode':
        self._final_module = ll.parse_bitcode(data)
        self._finalize_final_module()
        return self
    elif kind == 'object':
        object_code, shared_bitcode = data
        self.enable_object_caching()
        self._set_compiled_object(object_code)
        self._shared_module = ll.parse_bitcode(shared_bitcode)
        self._finalize_final_module()
        self._codegen._engine._load_defined_symbols(self._shared_module)
        return self
    else:
        raise ValueError('unsupported serialization kind %r' % (kind,))