import os
import sys
from llvmlite import ir
from numba.core import types, utils, config, cgutils, errors
from numba import gdb, gdb_init, gdb_breakpoint
from numba.core.extending import overload, intrinsic
def _confirm_gdb(need_ptrace_attach=True):
    """
    Set need_ptrace_attach to True/False to indicate whether the ptrace attach
    permission is needed for this gdb use case. Mode 0 (classic) or 1
    (restricted ptrace) is required if need_ptrace_attach is True. See:
    https://www.kernel.org/doc/Documentation/admin-guide/LSM/Yama.rst
    for details on the modes.
    """
    if not _unix_like:
        msg = 'gdb support is only available on unix-like systems'
        raise errors.NumbaRuntimeError(msg)
    gdbloc = config.GDB_BINARY
    if not (os.path.exists(gdbloc) and os.path.isfile(gdbloc)):
        msg = 'Is gdb present? Location specified (%s) does not exist. The gdb binary location can be set using Numba configuration, see: https://numba.readthedocs.io/en/stable/reference/envvars.html'
        raise RuntimeError(msg % config.GDB_BINARY)
    ptrace_scope_file = os.path.join(os.sep, 'proc', 'sys', 'kernel', 'yama', 'ptrace_scope')
    has_ptrace_scope = os.path.exists(ptrace_scope_file)
    if has_ptrace_scope:
        with open(ptrace_scope_file, 'rt') as f:
            value = f.readline().strip()
        if need_ptrace_attach and value not in ('0', '1'):
            msg = "gdb can launch but cannot attach to the executing program because ptrace permissions have been restricted at the system level by the Linux security module 'Yama'.\n\nDocumentation for this module and the security implications of making changes to its behaviour can be found in the Linux Kernel documentation https://www.kernel.org/doc/Documentation/admin-guide/LSM/Yama.rst\n\nDocumentation on how to adjust the behaviour of Yama on Ubuntu Linux with regards to 'ptrace_scope' can be found here https://wiki.ubuntu.com/Security/Features#ptrace."
            raise RuntimeError(msg)