import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
class _BlocksOutputBuffer:
    KB = 1024
    MB = 1024 * 1024
    BUFFER_BLOCK_SIZE = (32 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB, 8 * MB, 16 * MB, 16 * MB, 32 * MB, 32 * MB, 32 * MB, 32 * MB, 64 * MB, 64 * MB, 128 * MB, 128 * MB, 256 * MB)
    MEM_ERR_MSG = 'Unable to allocate output buffer.'

    def initAndGrow(self, out, max_length):
        self.max_length = max_length
        if 0 <= max_length < self.BUFFER_BLOCK_SIZE[0]:
            block_size = max_length
        else:
            block_size = self.BUFFER_BLOCK_SIZE[0]
        block = _new_nonzero('char[]', block_size)
        if block == ffi.NULL:
            raise MemoryError
        self.list = [block]
        self.allocated = block_size
        out.dst = block
        out.size = block_size
        out.pos = 0

    def grow(self, out):
        assert out.pos == out.size
        list_len = len(self.list)
        if list_len < len(self.BUFFER_BLOCK_SIZE):
            block_size = self.BUFFER_BLOCK_SIZE[list_len]
        else:
            block_size = self.BUFFER_BLOCK_SIZE[-1]
        if self.max_length >= 0:
            rest = self.max_length - self.allocated
            assert rest > 0
            if block_size > rest:
                block_size = rest
        b = _new_nonzero('char[]', block_size)
        if b == ffi.NULL:
            raise MemoryError(self.MEM_ERR_MSG)
        self.list.append(b)
        self.allocated += block_size
        out.dst = b
        out.size = block_size
        out.pos = 0

    def reachedMaxLength(self, out):
        assert out.pos == out.size
        return self.allocated == self.max_length

    def finish(self, out):
        if len(self.list) == 1 and out.pos == out.size or (len(self.list) == 2 and out.pos == 0):
            return bytes(ffi.buffer(self.list[0]))
        data_size = self.allocated - (out.size - out.pos)
        final = _new_nonzero('char[]', data_size)
        if final == ffi.NULL:
            raise MemoryError(self.MEM_ERR_MSG)
        posi = 0
        for block in self.list[:-1]:
            ffi.memmove(final + posi, block, len(block))
            posi += len(block)
        ffi.memmove(final + posi, self.list[-1], out.pos)
        return bytes(ffi.buffer(final))