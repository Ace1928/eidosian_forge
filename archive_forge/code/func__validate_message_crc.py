from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def _validate_message_crc(self):
    message_crc = self._parse_message_crc()
    message_bytes = self._parse_message_bytes()
    _validate_checksum(message_bytes, message_crc, crc=self._prelude.crc)
    return message_crc