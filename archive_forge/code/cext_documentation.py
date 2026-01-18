import struct
import google_crc32c.__config__  # type: ignore
from google_crc32c._crc32c import extend  # type: ignore
from google_crc32c._crc32c import value  # type: ignore
from google_crc32c._checksum import CommonChecksum
Update the checksum with a new chunk of data.

        Args:
            chunk (Optional[bytes]): a chunk of data used to extend
                the CRC32C checksum.
        