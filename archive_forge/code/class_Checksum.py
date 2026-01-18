import array
import struct
from google_crc32c._checksum import CommonChecksum
class Checksum(CommonChecksum):
    """Hashlib-alike helper for CRC32C operations.

    Args:
        initial_value (Optional[bytes]): the initial chunk of data from
            which the CRC32C checksum is computed.  Defaults to b''.
    """

    def __init__(self, initial_value=b''):
        self._crc = 0
        if initial_value != b'':
            self.update(initial_value)

    def update(self, data):
        """Update the checksum with a new chunk of data.

        Args:
            chunk (Optional[bytes]): a chunk of data used to extend
                the CRC32C checksum.
        """
        if type(data) != array.array or data.itemsize != 1:
            buffer = array.array('B', data)
        else:
            buffer = data
        self._crc = self._crc ^ 4294967295
        for b in buffer:
            table_poly = _TABLE[(b ^ self._crc) & 255]
            self._crc = table_poly ^ self._crc >> 8 & 4294967295
        self._crc = self._crc ^ 4294967295