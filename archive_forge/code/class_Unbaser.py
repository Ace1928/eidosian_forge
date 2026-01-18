import re
import string
import sys
from jsbeautifier.unpackers import UnpackingError
class Unbaser(object):
    """Functor for a given base. Will efficiently convert
    strings to natural numbers."""
    ALPHABET = {62: '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', 95: ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'}

    def __init__(self, base):
        self.base = base
        if 36 < base < 62:
            if not hasattr(self.ALPHABET, self.ALPHABET[62][:base]):
                self.ALPHABET[base] = self.ALPHABET[62][:base]
        if 2 <= base <= 36:
            self.unbase = lambda string: int(string, base)
        else:
            try:
                self.dictionary = dict(((cipher, index) for index, cipher in enumerate(self.ALPHABET[base])))
            except KeyError:
                raise TypeError('Unsupported base encoding.')
            self.unbase = self._dictunbaser

    def __call__(self, string):
        return self.unbase(string)

    def _dictunbaser(self, string):
        """Decodes a  value to an integer."""
        ret = 0
        for index, cipher in enumerate(string[::-1]):
            ret += self.base ** index * self.dictionary[cipher]
        return ret