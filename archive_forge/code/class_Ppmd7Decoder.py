import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
class Ppmd7Decoder(PpmdBaseDecoder):

    def __init__(self, max_order: int, mem_size: int):
        if mem_size > sys.maxsize:
            raise ValueError('Mem_size exceed to platform limit.')
        if _PPMD7_MIN_ORDER <= max_order <= _PPMD7_MAX_ORDER and _PPMD7_MIN_MEM_SIZE <= mem_size <= _PPMD7_MAX_MEM_SIZE:
            self.lock = Lock()
            self._init_common()
            self.ppmd = ffi.new('CPpmd7 *')
            self.rc = ffi.new('CPpmd7z_RangeDec *')
            self.threadInfo = ffi.new('ppmd_info *')
            self._eof = False
            self._finished = False
            self._needs_input = True
            lib.ppmd7_state_init(self.ppmd, max_order, mem_size, self._allocator)
        else:
            raise ValueError('PPMd wrong parameters.')

    def decode(self, data: Union[bytes, bytearray, memoryview], length: int) -> bytes:
        if not isinstance(length, int) or length < 0:
            raise PpmdError('Wrong length argument is specified. It should be positive integer.')
        self.lock.acquire()
        in_buf, use_input_buffer = self._setup_inBuffer(data)
        if not self.inited:
            lib.ppmd7_decompress_init(self.rc, self.reader, self.threadInfo, self._allocator)
            self.inited = True
        out, out_buf = self._setup_outBuffer()
        remaining: int = length
        out_size = 0
        while remaining > 0:
            size = min(out_buf.size, remaining)
            self.lock.release()
            out_size = lib.ppmd7_decompress(self.ppmd, self.rc, out_buf, in_buf, size, self.threadInfo)
            self.lock.acquire()
            if out_size == -2:
                self.lock.release()
                raise PpmdError('DecodeError.')
            if out_size == -1 or self.rc.Code == 0:
                self._eof = True
                self._needs_input = False
                break
            if out_size == 0:
                self._needs_input = True
                break
            if out_buf.pos == out_buf.size:
                out.grow(out_buf)
            remaining = remaining - out_size
        self._unconsumed_in(in_buf, use_input_buffer)
        res = out.finish(out_buf)
        self.lock.release()
        return res

    @property
    def needs_input(self):
        return self._needs_input

    @property
    def eof(self):
        return self._eof

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._finished:
            return
        self._finished = True
        lib.Ppmd7T_Free(self.ppmd, self.threadInfo, self._allocator)
        ffi.release(self.ppmd)
        ffi.release(self.rc)
        self._release()
        self._needs_input = False
        self.lock.release()