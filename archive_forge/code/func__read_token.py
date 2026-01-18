from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
from ._binary import o32le as o32
def _read_token(self):
    token = b''
    while len(token) <= 10:
        c = self.fp.read(1)
        if not c:
            break
        elif c in b_whitespace:
            if not token:
                continue
            break
        elif c == b'#':
            while self.fp.read(1) not in b'\r\n':
                pass
            continue
        token += c
    if not token:
        msg = 'Reached EOF while reading header'
        raise ValueError(msg)
    elif len(token) > 10:
        msg = f'Token too long in file header: {token.decode()}'
        raise ValueError(msg)
    return token