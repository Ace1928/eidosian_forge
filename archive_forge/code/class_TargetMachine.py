import os
from ctypes import (POINTER, c_char_p, c_longlong, c_int, c_size_t,
from llvmlite.binding import ffi
from llvmlite.binding.common import _decode_string, _encode_string
class TargetMachine(ffi.ObjectRef):

    def _dispose(self):
        self._capi.LLVMPY_DisposeTargetMachine(self)

    def add_analysis_passes(self, pm):
        """
        Register analysis passes for this target machine with a pass manager.
        """
        ffi.lib.LLVMPY_AddAnalysisPasses(self, pm)

    def set_asm_verbosity(self, verbose):
        """
        Set whether this target machine will emit assembly with human-readable
        comments describing control flow, debug information, and so on.
        """
        ffi.lib.LLVMPY_SetTargetMachineAsmVerbosity(self, verbose)

    def emit_object(self, module):
        """
        Represent the module as a code object, suitable for use with
        the platform's linker.  Returns a byte string.
        """
        return self._emit_to_memory(module, use_object=True)

    def emit_assembly(self, module):
        """
        Return the raw assembler of the module, as a string.

        llvm.initialize_native_asmprinter() must have been called first.
        """
        return _decode_string(self._emit_to_memory(module, use_object=False))

    def _emit_to_memory(self, module, use_object=False):
        """Returns bytes of object code of the module.

        Args
        ----
        use_object : bool
            Emit object code or (if False) emit assembly code.
        """
        with ffi.OutputString() as outerr:
            mb = ffi.lib.LLVMPY_TargetMachineEmitToMemory(self, module, int(use_object), outerr)
            if not mb:
                raise RuntimeError(str(outerr))
        bufptr = ffi.lib.LLVMPY_GetBufferStart(mb)
        bufsz = ffi.lib.LLVMPY_GetBufferSize(mb)
        try:
            return string_at(bufptr, bufsz)
        finally:
            ffi.lib.LLVMPY_DisposeMemoryBuffer(mb)

    @property
    def target_data(self):
        return TargetData(ffi.lib.LLVMPY_CreateTargetMachineData(self))

    @property
    def triple(self):
        with ffi.OutputString() as out:
            ffi.lib.LLVMPY_GetTargetMachineTriple(self, out)
            return str(out)