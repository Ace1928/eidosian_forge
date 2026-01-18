import re
from io import BytesIO
from .. import errors
def add_bytes_record(self, chunks, length, names):
    """Add a Bytes record with the given names.

        :param bytes: The chunks to insert.
        :param length: Total length of bytes in chunks
        :param names: The names to give the inserted bytes. Each name is
            a tuple of bytestrings. The bytestrings may not contain
            whitespace.
        :return: An offset, length tuple. The offset is the offset
            of the record within the container, and the length is the
            length of data that will need to be read to reconstitute the
            record. These offset and length can only be used with the pack
            interface - they might be offset by headers or other such details
            and thus are only suitable for use by a ContainerReader.
        """
    current_offset = self.current_offset
    if length < self._JOIN_WRITES_THRESHOLD:
        self.write_func(self._serialiser.bytes_header(length, names) + b''.join(chunks))
    else:
        self.write_func(self._serialiser.bytes_header(length, names))
        for chunk in chunks:
            self.write_func(chunk)
    self.records_written += 1
    return (current_offset, self.current_offset - current_offset)