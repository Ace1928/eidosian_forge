import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
def CustomAddressIterator(memory_map, condition):
    """
    Generator function that iterates through a memory map, filtering memory
    region blocks by any given condition.

    @type  memory_map: list( L{win32.MemoryBasicInformation} )
    @param memory_map: List of memory region information objects.
        Returned by L{Process.get_memory_map}.

    @type  condition: function
    @param condition: Callback function that returns C{True} if the memory
        block should be returned, or C{False} if it should be filtered.

    @rtype:  generator of L{win32.MemoryBasicInformation}
    @return: Generator object to iterate memory blocks.
    """
    for mbi in memory_map:
        if condition(mbi):
            address = mbi.BaseAddress
            max_addr = address + mbi.RegionSize
            while address < max_addr:
                yield address
                address = address + 1