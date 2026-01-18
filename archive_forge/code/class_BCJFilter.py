import struct
from typing import Union
class BCJFilter:
    _mask_to_allowed_number = [0, 1, 2, 4, 8, 9, 10, 12]
    _mask_to_bit_number = [0, 1, 2, 2, 3, 3, 3, 3]

    def __init__(self, func, readahead: int, is_encoder: bool, stream_size: int=0):
        self.is_encoder: bool = is_encoder
        self.prev_mask: int = 0
        self.prev_pos: int = -5
        self.current_position: int = 0
        self.stream_size: int = stream_size
        self.buffer = bytearray()
        self._method = func
        self._readahead = readahead

    def sparc_code(self) -> int:
        limit: int = len(self.buffer) - 4
        i: int = 0
        while i <= limit:
            if (self.buffer[i], self.buffer[i + 1] & 192) in [(64, 0), (127, 192)]:
                src = struct.unpack('>L', self.buffer[i:i + 4])[0] << 2
                distance: int = self.current_position + i
                if self.is_encoder:
                    dest = src + distance >> 2
                else:
                    dest = src - distance >> 2
                dest = 0 - (dest >> 22 & 1) << 22 & 1073741823 | dest & 4194303 | 1073741824
                self.buffer[i:i + 4] = struct.pack('>L', dest)
            i += 4
        self.current_position = i
        return i

    def ppc_code(self) -> int:
        limit: int = len(self.buffer) - 4
        i: int = 0
        while i <= limit:
            distance: int = self.current_position + i
            if self.buffer[i] & 252 == 72 and self.buffer[i + 3] & 3 == 1:
                src = struct.unpack('>L', self.buffer[i:i + 4])[0] & 67108860
                if self.is_encoder:
                    dest = src + distance
                else:
                    dest = src - distance
                dest = 72 << 24 | dest & 67108863 | 1
                self.buffer[i:i + 4] = struct.pack('>L', dest)
            i += 4
        self.current_position = i
        return i

    def _unpack_thumb(self, b: Union[bytearray, bytes, memoryview]) -> int:
        return (b[1] & 7) << 19 | b[0] << 11 | (b[3] & 7) << 8 | b[2]

    def _pack_thumb(self, val: int) -> bytes:
        b = bytes([val >> 11 & 255, 240 | val >> 19 & 7, val & 255, 248 | val >> 8 & 7])
        return b

    def armt_code(self) -> int:
        limit: int = len(self.buffer) - 4
        i: int = 0
        while i <= limit:
            if self.buffer[i + 1] & 248 == 240 and self.buffer[i + 3] & 248 == 248:
                src = self._unpack_thumb(self.buffer[i:i + 4]) << 1
                distance: int = self.current_position + i + 4
                if self.is_encoder:
                    dest = src + distance
                else:
                    dest = src - distance
                dest >>= 1
                self.buffer[i:i + 4] = self._pack_thumb(dest)
                i += 2
            i += 2
        self.current_position += i
        return i

    def arm_code(self) -> int:
        limit = len(self.buffer) - 4
        i = 0
        while i <= limit:
            if self.buffer[i + 3] == 235:
                src = struct.unpack('<L', self.buffer[i:i + 3] + b'\x00')[0] << 2
                distance = self.current_position + i + 8
                if self.is_encoder:
                    dest = src + distance >> 2
                else:
                    dest = src - distance >> 2
                self.buffer[i:i + 3] = struct.pack('<L', dest & 16777215)[:3]
            i += 4
        self.current_position += i
        return i

    def x86_code(self) -> int:
        """
        The code algorithm from liblzma/simple/x86.c
        It is slightly different from LZMA-SDK's bra86.c
        :return: buffer position
        """
        size: int = len(self.buffer)
        if size < 5:
            return 0
        if self.current_position - self.prev_pos > 5:
            self.prev_pos = self.current_position - 5
        view = memoryview(self.buffer)
        limit: int = size - 5
        buffer_pos: int = 0
        pos1: int = 0
        pos2: int = 0
        while buffer_pos <= limit:
            if pos1 >= 0:
                pos1 = self.buffer.find(233, buffer_pos, limit)
            if pos2 >= 0:
                pos2 = self.buffer.find(232, buffer_pos, limit)
            if pos1 < 0 and pos2 < 0:
                buffer_pos = limit + 1
                break
            elif pos1 < 0:
                buffer_pos = pos2
            elif pos2 < 0:
                buffer_pos = pos1
            else:
                buffer_pos = min(pos1, pos2)
            offset = self.current_position + buffer_pos - self.prev_pos
            self.prev_pos = self.current_position + buffer_pos
            if offset > 5:
                self.prev_mask = 0
            else:
                for i in range(offset):
                    self.prev_mask &= 119
                    self.prev_mask <<= 1
            if view[buffer_pos + 4] in [0, 255] and self.prev_mask >> 1 in self._mask_to_allowed_number:
                jump_target = self.buffer[buffer_pos + 1:buffer_pos + 5]
                src = struct.unpack('<L', jump_target)[0]
                distance = self.current_position + buffer_pos + 5
                idx = self._mask_to_bit_number[self.prev_mask >> 1]
                while True:
                    if self.is_encoder:
                        dest = src + distance & 4294967295
                    else:
                        dest = src - distance & 4294967295
                    if self.prev_mask == 0:
                        break
                    b = 255 & dest >> 24 - idx * 8
                    if not (b == 0 or b == 255):
                        break
                    src = dest ^ (1 << 32 - idx * 8) - 1 & 4294967295
                write_view = view[buffer_pos + 1:buffer_pos + 5]
                write_view[0:3] = (dest & 16777215).to_bytes(3, 'little')
                write_view[3:4] = [b'\x00', b'\xff'][dest >> 24 & 1]
                buffer_pos += 5
                self.prev_mask = 0
            else:
                buffer_pos += 1
                self.prev_mask |= 1
                if self.buffer[buffer_pos + 3] in [0, 255]:
                    self.prev_mask |= 16
        self.current_position += buffer_pos
        return buffer_pos

    def decode(self, data: Union[bytes, bytearray, memoryview], max_length: int=-1) -> bytes:
        self.buffer.extend(data)
        pos: int = self._method()
        if self.current_position > self.stream_size - self._readahead:
            offset: int = self.stream_size - self.current_position
            tmp = bytes(self.buffer[:pos + offset])
            self.current_position = self.stream_size
            self.buffer = bytearray()
        else:
            tmp = bytes(self.buffer[:pos])
            self.buffer = self.buffer[pos:]
        return tmp

    def encode(self, data: Union[bytes, bytearray, memoryview]) -> bytes:
        self.buffer.extend(data)
        pos: int = self._method()
        tmp = bytes(self.buffer[:pos])
        self.buffer = self.buffer[pos:]
        return tmp

    def flush(self) -> bytes:
        return bytes(self.buffer)