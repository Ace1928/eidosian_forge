from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def _parse_message_crc(self):
    prelude = self._prelude
    crc_bytes = self._data[prelude.payload_end:prelude.total_length]
    message_crc, _ = DecodeUtils.unpack_uint32(crc_bytes)
    return message_crc