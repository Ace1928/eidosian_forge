from __future__ import with_statement
from winappdbg.textio import HexDump
from winappdbg import win32
import ctypes
import warnings
def _import_dependencies(self):
    global capstone
    if capstone is None:
        import capstone
    self.__constants = {win32.ARCH_I386: (capstone.CS_ARCH_X86, capstone.CS_MODE_32), win32.ARCH_AMD64: (capstone.CS_ARCH_X86, capstone.CS_MODE_64), win32.ARCH_THUMB: (capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB), win32.ARCH_ARM: (capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM), win32.ARCH_ARM64: (capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)}
    try:
        self.__bug = not isinstance(capstone.cs_disasm_quick(capstone.CS_ARCH_X86, capstone.CS_MODE_32, '\x90', 1)[0], capstone.capstone.CsInsn)
    except AttributeError:
        self.__bug = False
    if self.__bug:
        warnings.warn('This version of the Capstone bindings is unstable, please upgrade to a newer one!', RuntimeWarning, stacklevel=4)