import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
class CrashDump(StaticClass):
    """
    Static functions for crash dumps.

    @type reg_template: str
    @cvar reg_template: Template for the L{dump_registers} method.
    """
    reg_template = {win32.ARCH_I386: 'eax=%(Eax).8x ebx=%(Ebx).8x ecx=%(Ecx).8x edx=%(Edx).8x esi=%(Esi).8x edi=%(Edi).8x\neip=%(Eip).8x esp=%(Esp).8x ebp=%(Ebp).8x %(efl_dump)s\ncs=%(SegCs).4x  ss=%(SegSs).4x  ds=%(SegDs).4x  es=%(SegEs).4x  fs=%(SegFs).4x  gs=%(SegGs).4x             efl=%(EFlags).8x\n', win32.ARCH_AMD64: 'rax=%(Rax).16x rbx=%(Rbx).16x rcx=%(Rcx).16x\nrdx=%(Rdx).16x rsi=%(Rsi).16x rdi=%(Rdi).16x\nrip=%(Rip).16x rsp=%(Rsp).16x rbp=%(Rbp).16x\n r8=%(R8).16x  r9=%(R9).16x r10=%(R10).16x\nr11=%(R11).16x r12=%(R12).16x r13=%(R13).16x\nr14=%(R14).16x r15=%(R15).16x\n%(efl_dump)s\ncs=%(SegCs).4x  ss=%(SegSs).4x  ds=%(SegDs).4x  es=%(SegEs).4x  fs=%(SegFs).4x  gs=%(SegGs).4x             efl=%(EFlags).8x\n'}

    @staticmethod
    def dump_flags(efl):
        """
        Dump the x86 processor flags.
        The output mimics that of the WinDBG debugger.
        Used by L{dump_registers}.

        @type  efl: int
        @param efl: Value of the eFlags register.

        @rtype:  str
        @return: Text suitable for logging.
        """
        if efl is None:
            return ''
        efl_dump = 'iopl=%1d' % ((efl & 12288) >> 12)
        if efl & 1048576:
            efl_dump += ' vip'
        else:
            efl_dump += '    '
        if efl & 524288:
            efl_dump += ' vif'
        else:
            efl_dump += '    '
        if efl & 2048:
            efl_dump += ' ov'
        else:
            efl_dump += ' no'
        if efl & 1024:
            efl_dump += ' dn'
        else:
            efl_dump += ' up'
        if efl & 512:
            efl_dump += ' ei'
        else:
            efl_dump += ' di'
        if efl & 128:
            efl_dump += ' ng'
        else:
            efl_dump += ' pl'
        if efl & 64:
            efl_dump += ' zr'
        else:
            efl_dump += ' nz'
        if efl & 16:
            efl_dump += ' ac'
        else:
            efl_dump += ' na'
        if efl & 4:
            efl_dump += ' pe'
        else:
            efl_dump += ' po'
        if efl & 1:
            efl_dump += ' cy'
        else:
            efl_dump += ' nc'
        return efl_dump

    @classmethod
    def dump_registers(cls, registers, arch=None):
        """
        Dump the x86/x64 processor register values.
        The output mimics that of the WinDBG debugger.

        @type  registers: dict( str S{->} int )
        @param registers: Dictionary mapping register names to their values.

        @type  arch: str
        @param arch: Architecture of the machine whose registers were dumped.
            Defaults to the current architecture.
            Currently only the following architectures are supported:
             - L{win32.ARCH_I386}
             - L{win32.ARCH_AMD64}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if registers is None:
            return ''
        if arch is None:
            if 'Eax' in registers:
                arch = win32.ARCH_I386
            elif 'Rax' in registers:
                arch = win32.ARCH_AMD64
            else:
                arch = 'Unknown'
        if arch not in cls.reg_template:
            msg = "Don't know how to dump the registers for architecture: %s"
            raise NotImplementedError(msg % arch)
        registers = registers.copy()
        registers['efl_dump'] = cls.dump_flags(registers['EFlags'])
        return cls.reg_template[arch] % registers

    @staticmethod
    def dump_registers_peek(registers, data, separator=' ', width=16):
        """
        Dump data pointed to by the given registers, if any.

        @type  registers: dict( str S{->} int )
        @param registers: Dictionary mapping register names to their values.
            This value is returned by L{Thread.get_context}.

        @type  data: dict( str S{->} str )
        @param data: Dictionary mapping register names to the data they point to.
            This value is returned by L{Thread.peek_pointers_in_registers}.

        @rtype:  str
        @return: Text suitable for logging.
        """
        if None in (registers, data):
            return ''
        names = compat.keys(data)
        names.sort()
        result = ''
        for reg_name in names:
            tag = reg_name.lower()
            dumped = HexDump.hexline(data[reg_name], separator, width)
            result += '%s -> %s\n' % (tag, dumped)
        return result

    @staticmethod
    def dump_data_peek(data, base=0, separator=' ', width=16, bits=None):
        """
        Dump data from pointers guessed within the given binary data.

        @type  data: str
        @param data: Dictionary mapping offsets to the data they point to.

        @type  base: int
        @param base: Base offset.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if data is None:
            return ''
        pointers = compat.keys(data)
        pointers.sort()
        result = ''
        for offset in pointers:
            dumped = HexDump.hexline(data[offset], separator, width)
            address = HexDump.address(base + offset, bits)
            result += '%s -> %s\n' % (address, dumped)
        return result

    @staticmethod
    def dump_stack_peek(data, separator=' ', width=16, arch=None):
        """
        Dump data from pointers guessed within the given stack dump.

        @type  data: str
        @param data: Dictionary mapping stack offsets to the data they point to.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each character.

        @type  width: int
        @param width:
            (Optional) Maximum number of characters to convert per text line.
            This value is also used for padding.

        @type  arch: str
        @param arch: Architecture of the machine whose registers were dumped.
            Defaults to the current architecture.

        @rtype:  str
        @return: Text suitable for logging.
        """
        if data is None:
            return ''
        if arch is None:
            arch = win32.arch
        pointers = compat.keys(data)
        pointers.sort()
        result = ''
        if pointers:
            if arch == win32.ARCH_I386:
                spreg = 'esp'
            elif arch == win32.ARCH_AMD64:
                spreg = 'rsp'
            else:
                spreg = 'STACK'
            tag_fmt = '[%s+0x%%.%dx]' % (spreg, len('%x' % pointers[-1]))
            for offset in pointers:
                dumped = HexDump.hexline(data[offset], separator, width)
                tag = tag_fmt % offset
                result += '%s -> %s\n' % (tag, dumped)
        return result

    @staticmethod
    def dump_stack_trace(stack_trace, bits=None):
        """
        Dump a stack trace, as returned by L{Thread.get_stack_trace} with the
        C{bUseLabels} parameter set to C{False}.

        @type  stack_trace: list( int, int, str )
        @param stack_trace: Stack trace as a list of tuples of
            ( return address, frame pointer, module filename )

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if not stack_trace:
            return ''
        table = Table()
        table.addRow('Frame', 'Origin', 'Module')
        for fp, ra, mod in stack_trace:
            fp_d = HexDump.address(fp, bits)
            ra_d = HexDump.address(ra, bits)
            table.addRow(fp_d, ra_d, mod)
        return table.getOutput()

    @staticmethod
    def dump_stack_trace_with_labels(stack_trace, bits=None):
        """
        Dump a stack trace,
        as returned by L{Thread.get_stack_trace_with_labels}.

        @type  stack_trace: list( int, int, str )
        @param stack_trace: Stack trace as a list of tuples of
            ( return address, frame pointer, module filename )

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if not stack_trace:
            return ''
        table = Table()
        table.addRow('Frame', 'Origin')
        for fp, label in stack_trace:
            table.addRow(HexDump.address(fp, bits), label)
        return table.getOutput()

    @staticmethod
    def dump_code(disassembly, pc=None, bLowercase=True, bits=None):
        """
        Dump a disassembly. Optionally mark where the program counter is.

        @type  disassembly: list of tuple( int, int, str, str )
        @param disassembly: Disassembly dump as returned by
            L{Process.disassemble} or L{Thread.disassemble_around_pc}.

        @type  pc: int
        @param pc: (Optional) Program counter.

        @type  bLowercase: bool
        @param bLowercase: (Optional) If C{True} convert the code to lowercase.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if not disassembly:
            return ''
        table = Table(sep=' | ')
        for addr, size, code, dump in disassembly:
            if bLowercase:
                code = code.lower()
            if addr == pc:
                addr = ' * %s' % HexDump.address(addr, bits)
            else:
                addr = '   %s' % HexDump.address(addr, bits)
            table.addRow(addr, dump, code)
        table.justify(1, 1)
        return table.getOutput()

    @staticmethod
    def dump_code_line(disassembly_line, bShowAddress=True, bShowDump=True, bLowercase=True, dwDumpWidth=None, dwCodeWidth=None, bits=None):
        """
        Dump a single line of code. To dump a block of code use L{dump_code}.

        @type  disassembly_line: tuple( int, int, str, str )
        @param disassembly_line: Single item of the list returned by
            L{Process.disassemble} or L{Thread.disassemble_around_pc}.

        @type  bShowAddress: bool
        @param bShowAddress: (Optional) If C{True} show the memory address.

        @type  bShowDump: bool
        @param bShowDump: (Optional) If C{True} show the hexadecimal dump.

        @type  bLowercase: bool
        @param bLowercase: (Optional) If C{True} convert the code to lowercase.

        @type  dwDumpWidth: int or None
        @param dwDumpWidth: (Optional) Width in characters of the hex dump.

        @type  dwCodeWidth: int or None
        @param dwCodeWidth: (Optional) Width in characters of the code.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if bits is None:
            address_size = HexDump.address_size
        else:
            address_size = bits / 4
        addr, size, code, dump = disassembly_line
        dump = dump.replace(' ', '')
        result = list()
        fmt = ''
        if bShowAddress:
            result.append(HexDump.address(addr, bits))
            fmt += '%%%ds:' % address_size
        if bShowDump:
            result.append(dump)
            if dwDumpWidth:
                fmt += ' %%-%ds' % dwDumpWidth
            else:
                fmt += ' %s'
        if bLowercase:
            code = code.lower()
        result.append(code)
        if dwCodeWidth:
            fmt += ' %%-%ds' % dwCodeWidth
        else:
            fmt += ' %s'
        return fmt % tuple(result)

    @staticmethod
    def dump_memory_map(memoryMap, mappedFilenames=None, bits=None):
        """
        Dump the memory map of a process. Optionally show the filenames for
        memory mapped files as well.

        @type  memoryMap: list( L{win32.MemoryBasicInformation} )
        @param memoryMap: Memory map returned by L{Process.get_memory_map}.

        @type  mappedFilenames: dict( int S{->} str )
        @param mappedFilenames: (Optional) Memory mapped filenames
            returned by L{Process.get_mapped_filenames}.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
        if not memoryMap:
            return ''
        table = Table()
        if mappedFilenames:
            table.addRow('Address', 'Size', 'State', 'Access', 'Type', 'File')
        else:
            table.addRow('Address', 'Size', 'State', 'Access', 'Type')
        for mbi in memoryMap:
            BaseAddress = HexDump.address(mbi.BaseAddress, bits)
            RegionSize = HexDump.address(mbi.RegionSize, bits)
            mbiState = mbi.State
            if mbiState == win32.MEM_RESERVE:
                State = 'Reserved'
            elif mbiState == win32.MEM_COMMIT:
                State = 'Commited'
            elif mbiState == win32.MEM_FREE:
                State = 'Free'
            else:
                State = 'Unknown'
            if mbiState != win32.MEM_COMMIT:
                Protect = ''
            else:
                mbiProtect = mbi.Protect
                if mbiProtect & win32.PAGE_NOACCESS:
                    Protect = '--- '
                elif mbiProtect & win32.PAGE_READONLY:
                    Protect = 'R-- '
                elif mbiProtect & win32.PAGE_READWRITE:
                    Protect = 'RW- '
                elif mbiProtect & win32.PAGE_WRITECOPY:
                    Protect = 'RC- '
                elif mbiProtect & win32.PAGE_EXECUTE:
                    Protect = '--X '
                elif mbiProtect & win32.PAGE_EXECUTE_READ:
                    Protect = 'R-X '
                elif mbiProtect & win32.PAGE_EXECUTE_READWRITE:
                    Protect = 'RWX '
                elif mbiProtect & win32.PAGE_EXECUTE_WRITECOPY:
                    Protect = 'RCX '
                else:
                    Protect = '??? '
                if mbiProtect & win32.PAGE_GUARD:
                    Protect += 'G'
                else:
                    Protect += '-'
                if mbiProtect & win32.PAGE_NOCACHE:
                    Protect += 'N'
                else:
                    Protect += '-'
                if mbiProtect & win32.PAGE_WRITECOMBINE:
                    Protect += 'W'
                else:
                    Protect += '-'
            mbiType = mbi.Type
            if mbiType == win32.MEM_IMAGE:
                Type = 'Image'
            elif mbiType == win32.MEM_MAPPED:
                Type = 'Mapped'
            elif mbiType == win32.MEM_PRIVATE:
                Type = 'Private'
            elif mbiType == 0:
                Type = ''
            else:
                Type = 'Unknown'
            if mappedFilenames:
                FileName = mappedFilenames.get(mbi.BaseAddress, '')
                table.addRow(BaseAddress, RegionSize, State, Protect, Type, FileName)
            else:
                table.addRow(BaseAddress, RegionSize, State, Protect, Type)
        return table.getOutput()