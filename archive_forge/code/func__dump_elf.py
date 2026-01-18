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
def _dump_elf(cls, buf):
    """
        Dump the symbol table of an ELF file.
        Needs pyelftools (https://github.com/eliben/pyelftools)
        """
    from elftools.elf.elffile import ELFFile
    from elftools.elf import descriptions
    from io import BytesIO
    f = ELFFile(BytesIO(buf))
    print('ELF file:')
    for sec in f.iter_sections():
        if sec['sh_type'] == 'SHT_SYMTAB':
            symbols = sorted(sec.iter_symbols(), key=lambda sym: sym.name)
            print('    symbols:')
            for sym in symbols:
                if not sym.name:
                    continue
                print('    - %r: size=%d, value=0x%x, type=%s, bind=%s' % (sym.name.decode(), sym['st_size'], sym['st_value'], descriptions.describe_symbol_type(sym['st_info']['type']), descriptions.describe_symbol_bind(sym['st_info']['bind'])))
    print()