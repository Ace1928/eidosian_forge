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
class AOTCPUCodegen(CPUCodegen):
    """
    A codegen implementation suitable for Ahead-Of-Time compilation
    (e.g. generation of object files).
    """
    _library_class = AOTCodeLibrary

    def __init__(self, module_name, cpu_name=None):
        self._cpu_name = cpu_name or ''
        CPUCodegen.__init__(self, module_name)

    def _customize_tm_options(self, options):
        cpu_name = self._cpu_name
        if cpu_name == 'host':
            cpu_name = self._get_host_cpu_name()
        options['cpu'] = cpu_name
        options['reloc'] = 'pic'
        options['codemodel'] = 'default'
        options['features'] = self._tm_features

    def _customize_tm_features(self):
        return ''

    def _add_module(self, module):
        pass