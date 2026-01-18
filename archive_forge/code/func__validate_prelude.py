from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def _validate_prelude(self, prelude):
    if prelude.headers_length > _MAX_HEADERS_LENGTH:
        raise InvalidHeadersLength(prelude.headers_length)
    if prelude.payload_length > _MAX_PAYLOAD_LENGTH:
        raise InvalidPayloadLength(prelude.payload_length)