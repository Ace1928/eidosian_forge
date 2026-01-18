from .binary import encode_bin, decode_bin
class BitStreamWriter(object):
    __slots__ = ['substream', 'buffer', 'pos']

    def __init__(self, substream):
        self.substream = substream
        self.buffer = []
        self.pos = 0

    def close(self):
        self.flush()

    def flush(self):
        bytes = decode_bin(''.join(self.buffer))
        self.substream.write(bytes)
        self.buffer = []
        self.pos = 0

    def tell(self):
        return self.substream.tell() + self.pos // 8

    def seek(self, pos, whence=0):
        self.flush()
        self.substream.seek(pos, whence)

    def write(self, data):
        if not data:
            return
        if type(data) is not str:
            raise TypeError('data must be a string, not %r' % (type(data),))
        self.buffer.append(data)