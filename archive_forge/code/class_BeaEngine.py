from __future__ import with_statement
from winappdbg.textio import HexDump
from winappdbg import win32
import ctypes
import warnings
class BeaEngine(Engine):
    """
    Integration with the BeaEngine disassembler by Beatrix.

    @see: U{https://sourceforge.net/projects/winappdbg/files/additional%20packages/BeaEngine/}
    """
    name = 'BeaEngine'
    desc = 'BeaEngine disassembler by Beatrix'
    url = 'https://sourceforge.net/projects/winappdbg/files/additional%20packages/BeaEngine/'
    supported = set((win32.ARCH_I386, win32.ARCH_AMD64))

    def _import_dependencies(self):
        global BeaEnginePython
        if BeaEnginePython is None:
            import BeaEnginePython

    def decode(self, address, code):
        addressof = ctypes.addressof
        buffer = ctypes.create_string_buffer(code)
        buffer_ptr = addressof(buffer)
        Instruction = BeaEnginePython.DISASM()
        Instruction.VirtualAddr = address
        Instruction.EIP = buffer_ptr
        Instruction.SecurityBlock = buffer_ptr + len(code)
        if self.arch == win32.ARCH_I386:
            Instruction.Archi = 0
        else:
            Instruction.Archi = 64
        Instruction.Options = BeaEnginePython.Tabulation + BeaEnginePython.NasmSyntax + BeaEnginePython.SuffixedNumeral + BeaEnginePython.ShowSegmentRegs
        result = []
        Disasm = BeaEnginePython.Disasm
        InstructionPtr = addressof(Instruction)
        hexdump = HexDump.hexadecimal
        append = result.append
        OUT_OF_BLOCK = BeaEnginePython.OUT_OF_BLOCK
        UNKNOWN_OPCODE = BeaEnginePython.UNKNOWN_OPCODE
        while True:
            offset = Instruction.EIP - buffer_ptr
            if offset >= len(code):
                break
            InstrLength = Disasm(InstructionPtr)
            if InstrLength == OUT_OF_BLOCK:
                break
            if InstrLength == UNKNOWN_OPCODE:
                char = '%.2X' % ord(buffer[offset])
                result.append((Instruction.VirtualAddr, 1, 'db %sh' % char, char))
                Instruction.VirtualAddr += 1
                Instruction.EIP += 1
            elif offset + InstrLength > len(code):
                for char in buffer[offset:offset + len(code)]:
                    char = '%.2X' % ord(char)
                    result.append((Instruction.VirtualAddr, 1, 'db %sh' % char, char))
                    Instruction.VirtualAddr += 1
                    Instruction.EIP += 1
            else:
                append((Instruction.VirtualAddr, InstrLength, Instruction.CompleteInstr.strip(), hexdump(buffer.raw[offset:offset + InstrLength])))
                Instruction.VirtualAddr += InstrLength
                Instruction.EIP += InstrLength
        return result