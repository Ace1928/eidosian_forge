import re
from io import BytesIO
from .. import errors
def _content_reader(self, max_length):
    if max_length is None:
        length_to_read = self._remaining_length
    else:
        length_to_read = min(max_length, self._remaining_length)
    self._remaining_length -= length_to_read
    bytes = self.reader_func(length_to_read)
    if len(bytes) != length_to_read:
        raise UnexpectedEndOfContainerError()
    return bytes