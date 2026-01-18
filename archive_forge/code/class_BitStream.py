from __future__ import annotations
from . import Image, ImageFile
from ._binary import i8
class BitStream:

    def __init__(self, fp):
        self.fp = fp
        self.bits = 0
        self.bitbuffer = 0

    def next(self):
        return i8(self.fp.read(1))

    def peek(self, bits):
        while self.bits < bits:
            c = self.next()
            if c < 0:
                self.bits = 0
                continue
            self.bitbuffer = (self.bitbuffer << 8) + c
            self.bits += 8
        return self.bitbuffer >> self.bits - bits & (1 << bits) - 1

    def skip(self, bits):
        while self.bits < bits:
            self.bitbuffer = (self.bitbuffer << 8) + i8(self.fp.read(1))
            self.bits += 8
        self.bits = self.bits - bits

    def read(self, bits):
        v = self.peek(bits)
        self.bits = self.bits - bits
        return v