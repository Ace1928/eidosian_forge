import base64
import binascii
import re
import string
import six
def ReverseChunk(value, size):
    """Break a string into chunks of a given size, starting at the rear.

  Like chunk, except the smallest chunk comes at the beginning.

  Args:
    value: The value to split.
    size: The maximum size of a chunk.

  Returns:
    Iterable over string slices of as close to the given size as possible.
    ReverseChunk('hello', 2) => 'h', 'el', 'lo'

  Raises:
    ValueError: If size <= 0.
  """
    if size <= 0:
        raise ValueError('invalid chunk size')

    def DoChunk():
        """Actually perform the chunking."""
        start = 0
        if len(value) % size:
            yield value[:len(value) % size]
            start = len(value) % size
        for chunk in Chunk(value, size, start=start):
            yield chunk
    return DoChunk()