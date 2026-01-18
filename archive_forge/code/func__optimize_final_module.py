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
def _optimize_final_module(self):
    """
        Internal: optimize this library's final module.
        """
    cheap_name = 'Module passes (cheap optimization for refprune)'
    with self._recorded_timings.record(cheap_name):
        self._codegen._mpm_cheap.run(self._final_module)
    if not config.LLVM_REFPRUNE_PASS:
        self._final_module = remove_redundant_nrt_refct(self._final_module)
    full_name = 'Module passes (full optimization)'
    with self._recorded_timings.record(full_name):
        self._codegen._mpm_full.run(self._final_module)