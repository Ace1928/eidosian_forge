from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def _parse_prelude(self):
    prelude_bytes = self._data[:_PRELUDE_LENGTH]
    raw_prelude, _ = DecodeUtils.unpack_prelude(prelude_bytes)
    prelude = MessagePrelude(*raw_prelude)
    self._validate_prelude(prelude)
    _validate_checksum(prelude_bytes[:_PRELUDE_LENGTH - 4], prelude.crc)
    return prelude