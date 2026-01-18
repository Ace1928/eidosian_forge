import warnings
import numpy as np
def _read_size(self, eof_ok=False):
    n = self._header_dtype.itemsize
    b = self._fp.read(n)
    if not b and eof_ok:
        raise FortranEOFError('End of file occurred at end of record')
    elif len(b) < n:
        raise FortranFormattingError('End of file in the middle of the record size')
    return int(np.frombuffer(b, dtype=self._header_dtype, count=1)[0])