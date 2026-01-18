import re
import struct
import binascii
def _85encode(b, chars, chars2, pad=False, foldnuls=False, foldspaces=False):
    if not isinstance(b, bytes_types):
        b = memoryview(b).tobytes()
    padding = -len(b) % 4
    if padding:
        b = b + b'\x00' * padding
    words = struct.Struct('!%dI' % (len(b) // 4)).unpack(b)
    chunks = [b'z' if foldnuls and (not word) else b'y' if foldspaces and word == 538976288 else chars2[word // 614125] + chars2[word // 85 % 7225] + chars[word % 85] for word in words]
    if padding and (not pad):
        if chunks[-1] == b'z':
            chunks[-1] = chars[0] * 5
        chunks[-1] = chunks[-1][:-padding]
    return b''.join(chunks)