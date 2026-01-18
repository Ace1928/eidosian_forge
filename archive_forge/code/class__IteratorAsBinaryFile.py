from requests.utils import super_len
from .multipart.encoder import CustomBytesIO, encode_with
class _IteratorAsBinaryFile(object):

    def __init__(self, iterator, encoding='utf-8'):
        self.iterator = iterator
        self.encoding = encoding
        self._buffer = CustomBytesIO()

    def _get_bytes(self):
        try:
            return encode_with(next(self.iterator), self.encoding)
        except StopIteration:
            return b''

    def _load_bytes(self, size):
        self._buffer.smart_truncate()
        amount_to_load = size - super_len(self._buffer)
        bytes_to_append = True
        while amount_to_load > 0 and bytes_to_append:
            bytes_to_append = self._get_bytes()
            amount_to_load -= self._buffer.append(bytes_to_append)

    def read(self, size=-1):
        size = int(size)
        if size == -1:
            return b''.join(self.iterator)
        self._load_bytes(size)
        return self._buffer.read(size)