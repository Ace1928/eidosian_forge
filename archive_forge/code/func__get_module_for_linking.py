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
def _get_module_for_linking(self):
    """
        Internal: get a LLVM module suitable for linking multiple times
        into another library.  Exported functions are made "linkonce_odr"
        to allow for multiple definitions, inlining, and removal of
        unused exports.

        See discussion in https://github.com/numba/numba/pull/890
        """
    self._ensure_finalized()
    if self._shared_module is not None:
        return self._shared_module
    mod = self._final_module
    to_fix = []
    nfuncs = 0
    for fn in mod.functions:
        nfuncs += 1
        if not fn.is_declaration and fn.linkage == ll.Linkage.external:
            to_fix.append(fn.name)
    if nfuncs == 0:
        raise RuntimeError('library unfit for linking: no available functions in %s' % (self,))
    if to_fix:
        mod = mod.clone()
        for name in to_fix:
            mod.get_function(name).linkage = 'linkonce_odr'
    self._shared_module = mod
    return mod