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
class CodeLibrary(metaclass=ABCMeta):
    """
    An interface for bundling LLVM code together and compiling it.
    It is tied to a *codegen* instance (e.g. JITCPUCodegen) that will
    determine how the LLVM code is transformed and linked together.
    """
    _finalized = False
    _object_caching_enabled = False
    _disable_inspection = False

    def __init__(self, codegen: 'CPUCodegen', name: str):
        self._codegen = codegen
        self._name = name
        ptc_name = f'{self.__class__.__name__}({self._name!r})'
        self._recorded_timings = PassTimingsCollection(ptc_name)
        self._dynamic_globals = []

    @property
    def has_dynamic_globals(self):
        self._ensure_finalized()
        return len(self._dynamic_globals) > 0

    @property
    def recorded_timings(self):
        return self._recorded_timings

    @property
    def codegen(self):
        """
        The codegen object owning this library.
        """
        return self._codegen

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return '<Library %r at 0x%x>' % (self.name, id(self))

    def _raise_if_finalized(self):
        if self._finalized:
            raise RuntimeError('operation impossible on finalized object %r' % (self,))

    def _ensure_finalized(self):
        if not self._finalized:
            self.finalize()

    def create_ir_module(self, name):
        """
        Create an LLVM IR module for use by this library.
        """
        self._raise_if_finalized()
        ir_module = self._codegen._create_empty_module(name)
        return ir_module

    @abstractmethod
    def add_linking_library(self, library):
        """
        Add a library for linking into this library, without losing
        the original library.
        """

    @abstractmethod
    def add_ir_module(self, ir_module):
        """
        Add an LLVM IR module's contents to this library.
        """

    @abstractmethod
    def finalize(self):
        """
        Finalize the library.  After this call, nothing can be added anymore.
        Finalization involves various stages of code optimization and
        linking.
        """

    @abstractmethod
    def get_function(self, name):
        """
        Return the function named ``name``.
        """

    @abstractmethod
    def get_llvm_str(self):
        """
        Get the human-readable form of the LLVM module.
        """

    @abstractmethod
    def get_asm_str(self):
        """
        Get the human-readable assembly.
        """

    def enable_object_caching(self):
        self._object_caching_enabled = True
        self._compiled_object = None
        self._compiled = False

    def _get_compiled_object(self):
        if not self._object_caching_enabled:
            raise ValueError('object caching not enabled in %s' % (self,))
        if self._compiled_object is None:
            raise RuntimeError('no compiled object yet for %s' % (self,))
        return self._compiled_object

    def _set_compiled_object(self, value):
        if not self._object_caching_enabled:
            raise ValueError('object caching not enabled in %s' % (self,))
        if self._compiled:
            raise ValueError('library already compiled: %s' % (self,))
        self._compiled_object = value
        self._disable_inspection = True