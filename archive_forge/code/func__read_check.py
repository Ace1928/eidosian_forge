import numpy
def _read_check(self):
    return numpy.frombuffer(self._read_exactly(self._header_length), dtype=self.ENDIAN + self.HEADER_PREC)[0]