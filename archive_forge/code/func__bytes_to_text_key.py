import struct
import zlib
from .static_tuple import StaticTuple
def _bytes_to_text_key(data):
    """Take a CHKInventory value string and return a (file_id, rev_id) tuple"""
    sections = data.split(b'\n')
    kind, file_id = sections[0].split(b': ')
    return (file_id, sections[3])