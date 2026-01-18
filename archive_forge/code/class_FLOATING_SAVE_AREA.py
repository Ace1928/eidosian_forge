from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
class FLOATING_SAVE_AREA(Structure):
    _pack_ = 1
    _fields_ = [('ControlWord', DWORD), ('StatusWord', DWORD), ('TagWord', DWORD), ('ErrorOffset', DWORD), ('ErrorSelector', DWORD), ('DataOffset', DWORD), ('DataSelector', DWORD), ('RegisterArea', BYTE * SIZE_OF_80387_REGISTERS), ('Cr0NpxState', DWORD)]
    _integer_members = ('ControlWord', 'StatusWord', 'TagWord', 'ErrorOffset', 'ErrorSelector', 'DataOffset', 'DataSelector', 'Cr0NpxState')

    @classmethod
    def from_dict(cls, fsa):
        """Instance a new structure from a Python dictionary."""
        fsa = dict(fsa)
        s = cls()
        for key in cls._integer_members:
            setattr(s, key, fsa.get(key))
        ra = fsa.get('RegisterArea', None)
        if ra is not None:
            for index in compat.xrange(0, SIZE_OF_80387_REGISTERS):
                s.RegisterArea[index] = ra[index]
        return s

    def to_dict(self):
        """Convert a structure into a Python dictionary."""
        fsa = dict()
        for key in self._integer_members:
            fsa[key] = getattr(self, key)
        ra = [self.RegisterArea[index] for index in compat.xrange(0, SIZE_OF_80387_REGISTERS)]
        ra = tuple(ra)
        fsa['RegisterArea'] = ra
        return fsa