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
def __is_control_flow(self):
    """
        Private method to tell if the instruction pointed to by the program
        counter is a control flow instruction.

        Currently only works for x86 and amd64 architectures.
        """
    jump_instructions = ('jmp', 'jecxz', 'jcxz', 'ja', 'jnbe', 'jae', 'jnb', 'jb', 'jnae', 'jbe', 'jna', 'jc', 'je', 'jz', 'jnc', 'jne', 'jnz', 'jnp', 'jpo', 'jp', 'jpe', 'jg', 'jnle', 'jge', 'jnl', 'jl', 'jnge', 'jle', 'jng', 'jno', 'jns', 'jo', 'js')
    call_instructions = ('call', 'ret', 'retn')
    loop_instructions = ('loop', 'loopz', 'loopnz', 'loope', 'loopne')
    control_flow_instructions = call_instructions + loop_instructions + jump_instructions
    isControlFlow = False
    instruction = None
    if self.pc is not None and self.faultDisasm:
        for disasm in self.faultDisasm:
            if disasm[0] == self.pc:
                instruction = disasm[2].lower().strip()
                break
    if instruction:
        for x in control_flow_instructions:
            if x in instruction:
                isControlFlow = True
                break
    return isControlFlow