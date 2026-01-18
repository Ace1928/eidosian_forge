from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
class CONTEXT(Structure):
    arch = ARCH_I386
    _pack_ = 1
    _fields_ = [('ContextFlags', DWORD), ('Dr0', DWORD), ('Dr1', DWORD), ('Dr2', DWORD), ('Dr3', DWORD), ('Dr6', DWORD), ('Dr7', DWORD), ('FloatSave', FLOATING_SAVE_AREA), ('SegGs', DWORD), ('SegFs', DWORD), ('SegEs', DWORD), ('SegDs', DWORD), ('Edi', DWORD), ('Esi', DWORD), ('Ebx', DWORD), ('Edx', DWORD), ('Ecx', DWORD), ('Eax', DWORD), ('Ebp', DWORD), ('Eip', DWORD), ('SegCs', DWORD), ('EFlags', DWORD), ('Esp', DWORD), ('SegSs', DWORD), ('ExtendedRegisters', BYTE * MAXIMUM_SUPPORTED_EXTENSION)]
    _ctx_debug = ('Dr0', 'Dr1', 'Dr2', 'Dr3', 'Dr6', 'Dr7')
    _ctx_segs = ('SegGs', 'SegFs', 'SegEs', 'SegDs')
    _ctx_int = ('Edi', 'Esi', 'Ebx', 'Edx', 'Ecx', 'Eax')
    _ctx_ctrl = ('Ebp', 'Eip', 'SegCs', 'EFlags', 'Esp', 'SegSs')

    @classmethod
    def from_dict(cls, ctx):
        """Instance a new structure from a Python dictionary."""
        ctx = Context(ctx)
        s = cls()
        ContextFlags = ctx['ContextFlags']
        setattr(s, 'ContextFlags', ContextFlags)
        if ContextFlags & CONTEXT_DEBUG_REGISTERS == CONTEXT_DEBUG_REGISTERS:
            for key in s._ctx_debug:
                setattr(s, key, ctx[key])
        if ContextFlags & CONTEXT_FLOATING_POINT == CONTEXT_FLOATING_POINT:
            fsa = ctx['FloatSave']
            s.FloatSave = FLOATING_SAVE_AREA.from_dict(fsa)
        if ContextFlags & CONTEXT_SEGMENTS == CONTEXT_SEGMENTS:
            for key in s._ctx_segs:
                setattr(s, key, ctx[key])
        if ContextFlags & CONTEXT_INTEGER == CONTEXT_INTEGER:
            for key in s._ctx_int:
                setattr(s, key, ctx[key])
        if ContextFlags & CONTEXT_CONTROL == CONTEXT_CONTROL:
            for key in s._ctx_ctrl:
                setattr(s, key, ctx[key])
        if ContextFlags & CONTEXT_EXTENDED_REGISTERS == CONTEXT_EXTENDED_REGISTERS:
            er = ctx['ExtendedRegisters']
            for index in compat.xrange(0, MAXIMUM_SUPPORTED_EXTENSION):
                s.ExtendedRegisters[index] = er[index]
        return s

    def to_dict(self):
        """Convert a structure into a Python native type."""
        ctx = Context()
        ContextFlags = self.ContextFlags
        ctx['ContextFlags'] = ContextFlags
        if ContextFlags & CONTEXT_DEBUG_REGISTERS == CONTEXT_DEBUG_REGISTERS:
            for key in self._ctx_debug:
                ctx[key] = getattr(self, key)
        if ContextFlags & CONTEXT_FLOATING_POINT == CONTEXT_FLOATING_POINT:
            ctx['FloatSave'] = self.FloatSave.to_dict()
        if ContextFlags & CONTEXT_SEGMENTS == CONTEXT_SEGMENTS:
            for key in self._ctx_segs:
                ctx[key] = getattr(self, key)
        if ContextFlags & CONTEXT_INTEGER == CONTEXT_INTEGER:
            for key in self._ctx_int:
                ctx[key] = getattr(self, key)
        if ContextFlags & CONTEXT_CONTROL == CONTEXT_CONTROL:
            for key in self._ctx_ctrl:
                ctx[key] = getattr(self, key)
        if ContextFlags & CONTEXT_EXTENDED_REGISTERS == CONTEXT_EXTENDED_REGISTERS:
            er = [self.ExtendedRegisters[index] for index in compat.xrange(0, MAXIMUM_SUPPORTED_EXTENSION)]
            er = tuple(er)
            ctx['ExtendedRegisters'] = er
        return ctx