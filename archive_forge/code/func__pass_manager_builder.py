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
def _pass_manager_builder(self, **kwargs):
    """
        Create a PassManagerBuilder.

        Note: a PassManagerBuilder seems good only for one use, so you
        should call this method each time you want to populate a module
        or function pass manager.  Otherwise some optimizations will be
        missed...
        """
    opt_level = kwargs.pop('opt', config.OPT)
    loop_vectorize = kwargs.pop('loop_vectorize', config.LOOP_VECTORIZE)
    slp_vectorize = kwargs.pop('slp_vectorize', config.SLP_VECTORIZE)
    pmb = create_pass_manager_builder(opt=opt_level, loop_vectorize=loop_vectorize, slp_vectorize=slp_vectorize, **kwargs)
    return pmb