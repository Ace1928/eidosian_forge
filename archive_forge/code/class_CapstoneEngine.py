from __future__ import with_statement
from winappdbg.textio import HexDump
from winappdbg import win32
import ctypes
import warnings
class CapstoneEngine(Engine):
    """
    Integration with the Capstone disassembler by Nguyen Anh Quynh.

    @see: U{http://www.capstone-engine.org/}
    """
    name = 'Capstone'
    desc = 'Capstone disassembler by Nguyen Anh Quynh'
    url = 'http://www.capstone-engine.org/'
    supported = set((win32.ARCH_I386, win32.ARCH_AMD64, win32.ARCH_THUMB, win32.ARCH_ARM, win32.ARCH_ARM64))

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

    def decode(self, address, code):
        arch, mode = self.__constants[self.arch]
        decoder = capstone.cs_disasm_quick
        if self.__bug:
            CsError = Exception
        else:
            CsError = capstone.CsError
        length = mnemonic = op_str = None
        result = []
        offset = 0
        while offset < len(code):
            instr = None
            try:
                instr = decoder(arch, mode, code[offset:offset + 16], address + offset, 1)[0]
            except IndexError:
                pass
            except CsError:
                pass
            if instr is not None:
                length = instr.size
                mnemonic = instr.mnemonic
                op_str = instr.op_str
                if op_str:
                    disasm = '%s %s' % (mnemonic, op_str)
                else:
                    disasm = mnemonic
                hexdump = HexDump.hexadecimal(code[offset:offset + length])
            else:
                if self.arch in (win32.ARCH_I386, win32.ARCH_AMD64):
                    length = 1
                else:
                    length = 4
                skipped = code[offset:offset + length]
                hexdump = HexDump.hexadecimal(skipped)
                if self.arch in (win32.ARCH_I386, win32.ARCH_AMD64):
                    mnemonic = 'db '
                else:
                    mnemonic = 'dcb '
                bytes = []
                for b in skipped:
                    if b.isalpha():
                        bytes.append("'%s'" % b)
                    else:
                        bytes.append('0x%x' % ord(b))
                op_str = ', '.join(bytes)
                disasm = mnemonic + op_str
            result.append((address + offset, length, disasm, hexdump))
            offset += length
        return result