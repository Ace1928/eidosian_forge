from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def _load_latest_dbghelp_dll():
    from os import getenv
    from os.path import join, exists
    program_files_location = getenv('ProgramFiles')
    if not program_files_location:
        program_files_location = 'C:\\Program Files'
    program_files_x86_location = getenv('ProgramFiles(x86)')
    if arch == ARCH_AMD64:
        if wow64:
            pathname = join(program_files_x86_location or program_files_location, 'Debugging Tools for Windows (x86)', 'dbghelp.dll')
        else:
            pathname = join(program_files_location, 'Debugging Tools for Windows (x64)', 'dbghelp.dll')
    elif arch == ARCH_I386:
        pathname = join(program_files_location, 'Debugging Tools for Windows (x86)', 'dbghelp.dll')
    else:
        pathname = None
    if pathname and exists(pathname):
        try:
            _dbghelp = ctypes.windll.LoadLibrary(pathname)
            ctypes.windll.dbghelp = _dbghelp
        except Exception:
            pass