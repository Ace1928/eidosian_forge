from winappdbg.win32.defines import *
def _get_bits():
    """
    Determines the current integer size in bits.

    This is useful to know if we're running in a 32 bits or a 64 bits machine.

    @rtype: int
    @return: Returns the size of L{SIZE_T} in bits.
    """
    return sizeof(SIZE_T) * 8