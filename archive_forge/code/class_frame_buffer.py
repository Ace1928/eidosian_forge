import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
class frame_buffer(object):
    _HEADER_MASK_INDEX = 5
    _HEADER_LENGTH_INDEX = 6

    def __init__(self, recv_fn, skip_utf8_validation):
        self.recv = recv_fn
        self.skip_utf8_validation = skip_utf8_validation
        self.recv_buffer = []
        self.clear()
        self.lock = Lock()

    def clear(self):
        self.header = None
        self.length = None
        self.mask = None

    def has_received_header(self):
        return self.header is None

    def recv_header(self):
        header = self.recv_strict(2)
        b1 = header[0]
        if six.PY2:
            b1 = ord(b1)
        fin = b1 >> 7 & 1
        rsv1 = b1 >> 6 & 1
        rsv2 = b1 >> 5 & 1
        rsv3 = b1 >> 4 & 1
        opcode = b1 & 15
        b2 = header[1]
        if six.PY2:
            b2 = ord(b2)
        has_mask = b2 >> 7 & 1
        length_bits = b2 & 127
        self.header = (fin, rsv1, rsv2, rsv3, opcode, has_mask, length_bits)

    def has_mask(self):
        if not self.header:
            return False
        return self.header[frame_buffer._HEADER_MASK_INDEX]

    def has_received_length(self):
        return self.length is None

    def recv_length(self):
        bits = self.header[frame_buffer._HEADER_LENGTH_INDEX]
        length_bits = bits & 127
        if length_bits == 126:
            v = self.recv_strict(2)
            self.length = struct.unpack('!H', v)[0]
        elif length_bits == 127:
            v = self.recv_strict(8)
            self.length = struct.unpack('!Q', v)[0]
        else:
            self.length = length_bits

    def has_received_mask(self):
        return self.mask is None

    def recv_mask(self):
        self.mask = self.recv_strict(4) if self.has_mask() else ''

    def recv_frame(self):
        with self.lock:
            if self.has_received_header():
                self.recv_header()
            fin, rsv1, rsv2, rsv3, opcode, has_mask, _ = self.header
            if self.has_received_length():
                self.recv_length()
            length = self.length
            if self.has_received_mask():
                self.recv_mask()
            mask = self.mask
            payload = self.recv_strict(length)
            if has_mask:
                payload = ABNF.mask(mask, payload)
            self.clear()
            frame = ABNF(fin, rsv1, rsv2, rsv3, opcode, has_mask, payload)
            frame.validate(self.skip_utf8_validation)
        return frame

    def recv_strict(self, bufsize):
        shortage = bufsize - sum((len(x) for x in self.recv_buffer))
        while shortage > 0:
            bytes_ = self.recv(min(16384, shortage))
            self.recv_buffer.append(bytes_)
            shortage -= len(bytes_)
        unified = six.b('').join(self.recv_buffer)
        if shortage == 0:
            self.recv_buffer = []
            return unified
        else:
            self.recv_buffer = [unified[bufsize:]]
            return unified[:bufsize]