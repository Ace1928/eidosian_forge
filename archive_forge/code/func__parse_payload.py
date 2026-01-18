from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def _parse_payload(self):
    prelude = self._prelude
    payload_bytes = self._data[prelude.headers_end:prelude.payload_end]
    return payload_bytes