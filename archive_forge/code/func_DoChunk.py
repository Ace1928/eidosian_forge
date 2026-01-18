import base64
import binascii
import re
import string
import six
def DoChunk():
    """Actually perform the chunking."""
    start = 0
    if len(value) % size:
        yield value[:len(value) % size]
        start = len(value) % size
    for chunk in Chunk(value, size, start=start):
        yield chunk