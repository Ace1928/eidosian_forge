import numpy
def _write_check(self, number_of_bytes):
    """Write the header for the given number of bytes"""
    self.write(numpy.array(number_of_bytes, dtype=self.ENDIAN + self.HEADER_PREC).tostring())