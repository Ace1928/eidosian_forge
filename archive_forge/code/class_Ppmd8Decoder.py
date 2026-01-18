import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
class Ppmd8Decoder(PpmdBaseDecoder):

    def __init__(self, max_order: int, mem_size: int, restore_method=PPMD8_RESTORE_METHOD_RESTART):
        self._init_common()
        self.ppmd = ffi.new('CPpmd8 *')
        self.threadInfo = ffi.new('ppmd_info *')
        lib.Ppmd8_Construct(self.ppmd)
        lib.Ppmd8_Alloc(self.ppmd, mem_size, self._allocator)
        lib.Ppmd8_Init(self.ppmd, max_order, restore_method)
        self._inited = False
        self._eof = False
        self._needs_input = True
        self._finished = False

    def _init2(self):
        lib.ppmd8_decompress_init(self.ppmd, self.reader, self.threadInfo, self._allocator)
        lib.Ppmd8_RangeDec_Init(self.ppmd)

    def decode(self, data: Union[bytes, bytearray, memoryview], length: int=-1):
        if not isinstance(length, int):
            raise PpmdError('Wrong length argument is specified.')
        self.lock.acquire()
        in_buf, use_input_buffer = self._setup_inBuffer(data)
        out, out_buf = self._setup_outBuffer()
        self.threadInfo.out = out_buf
        if not self._inited:
            self._inited = True
            self._init2()
        if length < 0:
            length = 2147483647
        while True:
            if out_buf.pos == length:
                break
            if out_buf.pos == out_buf.size:
                out.grow(out_buf)
            self.lock.release()
            size = lib.ppmd8_decompress(self.ppmd, out_buf, in_buf, length, self.threadInfo)
            self.lock.acquire()
            if size == -1:
                self._eof = True
                self._needs_input = False
                res = out.finish(out_buf)
                self.lock.release()
                self._free()
                return res
            elif size == -2:
                raise ValueError('Corrupted archive data.')
            if in_buf.pos == in_buf.size:
                break
        self._unconsumed_in(in_buf, use_input_buffer)
        if self._eof:
            self._needs_input = False
        elif self._input_buffer_size == 0 or self._in_begin == self._in_end:
            self._needs_input = True
        else:
            self._needs_input = False
        res = out.finish(out_buf)
        self.lock.release()
        return res

    def _free(self):
        if self._finished:
            return
        self._finished = True
        lib.Ppmd8T_Free(self.ppmd, self.threadInfo, self._allocator)
        ffi.release(self.ppmd)
        self._release()

    @property
    def needs_input(self):
        return self._needs_input

    @property
    def eof(self):
        return self._eof

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._free()