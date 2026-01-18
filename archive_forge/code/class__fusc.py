import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
class _fusc:

    def __init__(self, k, n):
        assert k, 'Argument k should be a non empty string'
        self._k = k
        self._klen = len(k)
        self._n = int(n) or 7

    def encrypt(self, s):
        return self.__rotate(asciiBase85Encode(''.join(map(chr, self.__fusc(list(map(ord, s)))))), self._n)

    def decrypt(self, s):
        return ''.join(map(chr, self.__fusc(list(map(ord, asciiBase85Decode(self.__rotate(s, -self._n)))))))

    def __rotate(self, s, n):
        l = len(s)
        if n < 0:
            n = l + n
        n %= l
        if not n:
            return s
        return s[-n:] + s[:l - n]

    def __fusc(self, s):
        slen = len(s)
        return list(map(lambda x, y: x ^ y, s, list(map(ord, ((int(slen / self._klen) + 1) * self._k)[:slen]))))