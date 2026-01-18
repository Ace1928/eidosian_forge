import re
import itertools
class QRStructAppend(QR):
    mode = 3
    lengthbits = (0, 0, 0)

    def __init__(self, part, total, parity):
        if not 0 < part <= 16:
            raise ValueError('part out of range [1,16]')
        if not 0 < total <= 16:
            raise ValueError('total out of range [1,16]')
        self.part = part
        self.total = total
        self.parity = parity

    def write(self, buffer, version):
        self.write_header(buffer, version)
        buffer.put(self.part, 4)
        buffer.put(self.total, 4)
        buffer.put(self.parity, 8)