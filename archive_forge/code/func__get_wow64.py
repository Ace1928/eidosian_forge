from winappdbg.win32.defines import *
def _get_wow64():
    """
    Determines if the current process is running in Windows-On-Windows 64 bits.

    @rtype:  bool
    @return: C{True} of the current process is a 32 bit program running in a
        64 bit version of Windows, C{False} if it's either a 32 bit program
        in a 32 bit Windows or a 64 bit program in a 64 bit Windows.
    """
    if bits == 64:
        wow64 = False
    else:
        try:
            wow64 = IsWow64Process(GetCurrentProcess())
        except Exception:
            wow64 = False
    return wow64