import re
import itertools
class QRECI(QR):
    mode = 7
    lengthbits = (0, 0, 0)

    def __init__(self, data):
        if not 0 < data < 999999:
            raise ValueError('ECI out of range')
        self.data = data

    def write(self, buffer, version):
        self.write_header(buffer, version)
        if self.data <= 127:
            buffer.put(self.data, 8)
        elif self.data <= 16383:
            buffer.put(self.data | 32768, 16)
        elif self.data <= 2097151:
            buffer.put(self.data | 12582912, 24)