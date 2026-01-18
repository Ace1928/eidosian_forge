from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations
import sys
import os
import time
import zlib
import warnings
def __is_block_data_move(self):
    """
        Private method to tell if the instruction pointed to by the program
        counter is a block data move instruction.

        Currently only works for x86 and amd64 architectures.
        """
    block_data_move_instructions = ('movs', 'stos', 'lods')
    isBlockDataMove = False
    instruction = None
    if self.pc is not None and self.faultDisasm:
        for disasm in self.faultDisasm:
            if disasm[0] == self.pc:
                instruction = disasm[2].lower().strip()
                break
    if instruction:
        for x in block_data_move_instructions:
            if x in instruction:
                isBlockDataMove = True
                break
    return isBlockDataMove