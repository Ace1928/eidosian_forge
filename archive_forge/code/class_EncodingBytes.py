from __future__ import absolute_import, division, unicode_literals
from six import text_type
from six.moves import http_client, urllib
import codecs
import re
from io import BytesIO, StringIO
from tensorboard._vendor import webencodings
from .constants import EOF, spaceCharacters, asciiLetters, asciiUppercase
from .constants import _ReparseException
from . import _utils
class EncodingBytes(bytes):
    """String-like object with an associated position and various extra methods
    If the position is ever greater than the string length then an exception is
    raised"""

    def __new__(self, value):
        assert isinstance(value, bytes)
        return bytes.__new__(self, value.lower())

    def __init__(self, value):
        self._position = -1

    def __iter__(self):
        return self

    def __next__(self):
        p = self._position = self._position + 1
        if p >= len(self):
            raise StopIteration
        elif p < 0:
            raise TypeError
        return self[p:p + 1]

    def next(self):
        return self.__next__()

    def previous(self):
        p = self._position
        if p >= len(self):
            raise StopIteration
        elif p < 0:
            raise TypeError
        self._position = p = p - 1
        return self[p:p + 1]

    def setPosition(self, position):
        if self._position >= len(self):
            raise StopIteration
        self._position = position

    def getPosition(self):
        if self._position >= len(self):
            raise StopIteration
        if self._position >= 0:
            return self._position
        else:
            return None
    position = property(getPosition, setPosition)

    def getCurrentByte(self):
        return self[self.position:self.position + 1]
    currentByte = property(getCurrentByte)

    def skip(self, chars=spaceCharactersBytes):
        """Skip past a list of characters"""
        p = self.position
        while p < len(self):
            c = self[p:p + 1]
            if c not in chars:
                self._position = p
                return c
            p += 1
        self._position = p
        return None

    def skipUntil(self, chars):
        p = self.position
        while p < len(self):
            c = self[p:p + 1]
            if c in chars:
                self._position = p
                return c
            p += 1
        self._position = p
        return None

    def matchBytes(self, bytes):
        """Look for a sequence of bytes at the start of a string. If the bytes
        are found return True and advance the position to the byte after the
        match. Otherwise return False and leave the position alone"""
        rv = self.startswith(bytes, self.position)
        if rv:
            self.position += len(bytes)
        return rv

    def jumpTo(self, bytes):
        """Look for the next sequence of bytes matching a given sequence. If
        a match is found advance the position to the last byte of the match"""
        try:
            self._position = self.index(bytes, self.position) + len(bytes) - 1
        except ValueError:
            raise StopIteration
        return True