import struct, sys, time, os
import zlib
import builtins
import io
import _compression
def _init_write(self, filename):
    self.name = filename
    self.crc = zlib.crc32(b'')
    self.size = 0
    self.writebuf = []
    self.bufsize = 0
    self.offset = 0