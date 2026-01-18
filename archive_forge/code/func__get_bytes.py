from requests.utils import super_len
from .multipart.encoder import CustomBytesIO, encode_with
def _get_bytes(self):
    try:
        return encode_with(next(self.iterator), self.encoding)
    except StopIteration:
        return b''