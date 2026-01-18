from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_AMD64
from winappdbg.win32 import context_i386
class XMM_SAVE_AREA32(Structure):
    _pack_ = 1
    _fields_ = [('ControlWord', WORD), ('StatusWord', WORD), ('TagWord', BYTE), ('Reserved1', BYTE), ('ErrorOpcode', WORD), ('ErrorOffset', DWORD), ('ErrorSelector', WORD), ('Reserved2', WORD), ('DataOffset', DWORD), ('DataSelector', WORD), ('Reserved3', WORD), ('MxCsr', DWORD), ('MxCsr_Mask', DWORD), ('FloatRegisters', M128A * 8), ('XmmRegisters', M128A * 16), ('Reserved4', BYTE * 96)]

    def from_dict(self):
        raise NotImplementedError()

    def to_dict(self):
        d = dict()
        for name, type in self._fields_:
            if name in ('FloatRegisters', 'XmmRegisters'):
                d[name] = tuple([x.LowPart + (x.HighPart << 64) for x in getattr(self, name)])
            elif name == 'Reserved4':
                d[name] = tuple([chr(x) for x in getattr(self, name)])
            else:
                d[name] = getattr(self, name)
        return d