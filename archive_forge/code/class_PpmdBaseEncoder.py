import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
class PpmdBaseEncoder:

    def __init__(self):
        pass

    def _init_common(self):
        self.lock = Lock()
        self.closed = False
        self.flushed = False
        self.writer = ffi.new('BufferWriter *')
        self._allocator = ffi.new('IAlloc *')
        self._allocator.Alloc = lib.raw_alloc
        self._allocator.Free = lib.raw_free

    def _setup_inBuffer(self, data):
        in_buf = _new_nonzero('InBuffer *')
        if in_buf == ffi.NULL:
            raise MemoryError
        in_buf.src = ffi.from_buffer(data)
        in_buf.size = len(data)
        in_buf.pos = 0
        return in_buf

    def _setup_outBuffer(self):
        out_buf = _new_nonzero('OutBuffer *')
        self.writer.outBuffer = out_buf
        if out_buf == ffi.NULL:
            raise MemoryError
        out = _BlocksOutputBuffer()
        out.initAndGrow(out_buf, -1)
        return (out, out_buf)

    def encode(self, data) -> bytes:
        return b''

    def flush(self) -> bytes:
        return b''

    def _release(self):
        ffi.release(self._allocator)
        ffi.release(self.writer)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if not self.flushed:
            self.flush()