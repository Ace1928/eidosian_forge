import re
from io import BytesIO
from .. import errors
class ReadVFile:
    """Adapt a readv result iterator to a file like protocol.

    The readv result must support the iterator protocol returning (offset,
    data_bytes) pairs.
    """

    def __init__(self, readv_result):
        """Construct a new ReadVFile wrapper.

        :seealso: make_readv_reader

        :param readv_result: the most recent readv result - list or generator
        """
        readv_result = iter(readv_result)
        self.readv_result = readv_result
        self._string = None

    def _next(self):
        if self._string is None or self._string.tell() == self._string_length:
            offset, data = next(self.readv_result)
            self._string_length = len(data)
            self._string = BytesIO(data)

    def read(self, length):
        self._next()
        result = self._string.read(length)
        if len(result) < length:
            raise errors.BzrError('wanted %d bytes but next hunk only contains %d: %r...' % (length, len(result), result[:20]))
        return result

    def readline(self):
        """Note that readline will not cross readv segments."""
        self._next()
        result = self._string.readline()
        if self._string.tell() == self._string_length and result[-1:] != b'\n':
            raise errors.BzrError('short readline in the readvfile hunk: %r' % (result,))
        return result