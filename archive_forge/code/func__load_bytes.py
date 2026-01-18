from requests.utils import super_len
from .multipart.encoder import CustomBytesIO, encode_with
def _load_bytes(self, size):
    self._buffer.smart_truncate()
    amount_to_load = size - super_len(self._buffer)
    bytes_to_append = True
    while amount_to_load > 0 and bytes_to_append:
        bytes_to_append = self._get_bytes()
        amount_to_load -= self._buffer.append(bytes_to_append)