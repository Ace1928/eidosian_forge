import struct
class CommonChecksum(object):
    """Hashlib-alike helper for CRC32C operations.

    This class should not be used directly and requires an update implementation.

    Args:
        initial_value (Optional[bytes]): the initial chunk of data from
            which the CRC32C checksum is computed.  Defaults to b''.
    """
    __slots__ = ()

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
        raise NotImplemented()

    def digest(self):
        """Big-endian order, per RFC 4960.

        See: https://cloud.google.com/storage/docs/json_api/v1/objects#crc32c

        Returns:
            bytes: An eight-byte digest string.
        """
        return struct.pack('>L', self._crc)

    def hexdigest(self):
        """Like :meth:`digest` except returns as a bytestring of double length.

        Returns
            bytes: A sixteen byte digest string, contaiing only hex digits.
        """
        return '{:08x}'.format(self._crc).encode('ascii')

    def copy(self):
        """Create another checksum with the same CRC32C value.

        Returns:
            Checksum: the new instance.
        """
        clone = self.__class__()
        clone._crc = self._crc
        return clone

    def consume(self, stream, chunksize):
        """Consume chunks from a stream, extending our CRC32 checksum.

        Args:
            stream (BinaryIO): the stream to consume.
            chunksize (int): the size of the read to perform

        Returns:
            Generator[bytes, None, None]: Iterable of the chunks read from the
            stream.
        """
        while True:
            chunk = stream.read(chunksize)
            if not chunk:
                break
            self.update(chunk)
            yield chunk