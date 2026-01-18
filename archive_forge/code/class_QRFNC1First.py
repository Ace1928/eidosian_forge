import re
import itertools
class QRFNC1First(QR):
    mode = 5
    lengthbits = (0, 0, 0)

    def __init__(self):
        pass

    def write(self, buffer, version):
        self.write_header(buffer, version)