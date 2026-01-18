from __future__ import annotations
import itertools
import logging
import re
import struct
import warnings
import zlib
from enum import IntEnum
from . import Image, ImageChops, ImageFile, ImagePalette, ImageSequence
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from ._binary import o32be as o32
class PngInfo:
    """
    PNG chunk container (for use with save(pnginfo=))

    """

    def __init__(self):
        self.chunks = []

    def add(self, cid, data, after_idat=False):
        """Appends an arbitrary chunk. Use with caution.

        :param cid: a byte string, 4 bytes long.
        :param data: a byte string of the encoded data
        :param after_idat: for use with private chunks. Whether the chunk
                           should be written after IDAT

        """
        chunk = [cid, data]
        if after_idat:
            chunk.append(True)
        self.chunks.append(tuple(chunk))

    def add_itxt(self, key, value, lang='', tkey='', zip=False):
        """Appends an iTXt chunk.

        :param key: latin-1 encodable text key name
        :param value: value for this key
        :param lang: language code
        :param tkey: UTF-8 version of the key name
        :param zip: compression flag

        """
        if not isinstance(key, bytes):
            key = key.encode('latin-1', 'strict')
        if not isinstance(value, bytes):
            value = value.encode('utf-8', 'strict')
        if not isinstance(lang, bytes):
            lang = lang.encode('utf-8', 'strict')
        if not isinstance(tkey, bytes):
            tkey = tkey.encode('utf-8', 'strict')
        if zip:
            self.add(b'iTXt', key + b'\x00\x01\x00' + lang + b'\x00' + tkey + b'\x00' + zlib.compress(value))
        else:
            self.add(b'iTXt', key + b'\x00\x00\x00' + lang + b'\x00' + tkey + b'\x00' + value)

    def add_text(self, key, value, zip=False):
        """Appends a text chunk.

        :param key: latin-1 encodable text key name
        :param value: value for this key, text or an
           :py:class:`PIL.PngImagePlugin.iTXt` instance
        :param zip: compression flag

        """
        if isinstance(value, iTXt):
            return self.add_itxt(key, value, value.lang, value.tkey, zip=zip)
        if not isinstance(value, bytes):
            try:
                value = value.encode('latin-1', 'strict')
            except UnicodeError:
                return self.add_itxt(key, value, zip=zip)
        if not isinstance(key, bytes):
            key = key.encode('latin-1', 'strict')
        if zip:
            self.add(b'zTXt', key + b'\x00\x00' + zlib.compress(value))
        else:
            self.add(b'tEXt', key + b'\x00' + value)