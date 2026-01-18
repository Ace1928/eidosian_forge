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
def _module_pass_manager(self, **kwargs):
    pm = ll.create_module_pass_manager()
    pm.add_target_library_info(ll.get_process_triple())
    self._tm.add_analysis_passes(pm)
    cost = kwargs.pop('cost', None)
    with self._pass_manager_builder(**kwargs) as pmb:
        pmb.populate(pm)
    if cost is not None and cost == 'cheap' and (config.OPT != 0):
        pm.add_loop_rotate_pass()
        if ll.llvm_version_info[0] < 12:
            pm.add_licm_pass()
            pm.add_cfg_simplification_pass()
        else:
            pm.add_instruction_combining_pass()
            pm.add_jump_threading_pass()
    if config.LLVM_REFPRUNE_PASS:
        pm.add_refprune_pass(_parse_refprune_flags())
    return pm